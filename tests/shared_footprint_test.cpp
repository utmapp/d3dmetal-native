/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Memory audit of every shared-resource path we intercept: does any D3D entry
 * point end up backing ONE logical resource with TWO allocations?
 *
 * The sharing machinery substitutes its own shared-memory-backed Metal object
 * for the one D3DMetal was about to create, so a shared resource should cost
 * the same as an unshared one. A path where the substitution is additive
 * rather than substitutive — the framework allocates and we allocate too —
 * costs twice the memory, and no other test in the suite can see it: they all
 * check that the bytes are correct, never how many allocations hold them.
 *
 * == How a case is measured ==
 *
 * Each case HOLDS N resources at once and reports two numbers per byte of
 * resource: how many bytes were RESERVED, and how many the machine is actually
 * HOLDING. See "Reserved vs resident" below for why both exist and which one
 * the verdict uses.
 *
 * RESERVED comes from MTLDevice.currentAllocatedSize (see metal_alloc_size.mm),
 * read twice — at payload P and 2P with N fixed — with the multiplier taken as
 * the SLOPE between the two —
 * see run_case for why a single ratio cannot separate a multiplier from a
 * fixed overhead, and what the "fixed" column is.
 *
 * Holding N rather than measuring one at a time keeps the slope out of the
 * noise: D3DMetal suballocates unshared resources out of a pool it grows in
 * 256 MiB chunks, so a single create usually reports +0 and occasionally
 * +256 MiB. Keep DMN_FP_N * DMN_FP_MB comfortably above 1 GiB.
 *
 * == Why these two metrics ==
 *
 * The task's phys_footprint is unusable here: a GPU-only write to shm-backed
 * memory (the shape of every substituted shared resource) is charged NOTHING
 * to it, it charges per MAPPING rather than per physical page (an aliased
 * import reads double), and it lags releases into the next case's baseline.
 *
 * RESERVED therefore comes from MTLDevice.currentAllocatedSize, which is
 * immediate, exact, symmetric on release, and counts the substituted backings
 * as well as the framework's own allocations. RESIDENT comes from host-wide
 * page counts (wired+active+compressed), the only accounting on macOS that
 * counts each physical page exactly once; GPU-written pages land in
 * wire_count for shm-backed and private allocations alike.
 *
 * The verdict is on RESIDENT, because pages the machine is actually holding
 * are what slow everything else down; a reservation nothing writes stays
 * virtual, while one referenced by submitted work becomes real memory (a GPU
 * write commits a whole Metal allocation at once). Two properties of the page
 * accounting shape how it is measured:
 *
 *  1. It LAGS GPU completion by up to about a second, so mark_resident() waits
 *     for the number to plateau instead of reading it once at fence-retire.
 *  2. Freed pages are not reclaimed until something needs them, so within one
 *     process a case's baseline still holds the previous case's released
 *     memory and the two cancel. Only process exit returns pages
 *     unconditionally, so each case's resident figure is measured in a fresh
 *     child (measure_resident_child).
 *
 * "CreateHeap only" and "heap_from_fd only" isolate a heap's own cost from
 * that of the resources placed in it.
 *
 * == What fails the test ==
 *
 * A sharing path holding at or above DMN_FP_MAX_RESIDENT (default 1.6) bytes
 * of RESIDENT memory per byte of resource FAILS (RAM-DOUBLE), as does one that
 * produces more distinct shm objects than there are resources (COPIES — it is
 * not aliasing). A path that over-RESERVES but holds ~1x resident is reported
 * as `reserve2x` and does not fail: address space and video-memory budget are
 * worth knowing about, but they are not what slows a machine down.
 *
 * Unshared controls are measured but never judged: they are the reference for
 * what the same create costs without us in the path. A create that asked for
 * sharing and silently got a private resource (UNSHARED) is a capability gap
 * rather than a memory bug, so it warns; DMN_FP_STRICT=1 fails on that too.
 * DMN_FP_REPORT=1 restores report-only behaviour (always exit 0).
 *
 * Env: DMN_FP_MB=<n>      payload per resource in MiB (default 64)
 *      DMN_FP_N=<n>       resources held at once (default 16)
 *      DMN_FP_MAX=<f>     reserved ratio reported as reserve2x (default 1.75)
 *      DMN_FP_MAX_RESIDENT=<f>  failing resident ratio (default 1.6)
 *      DMN_FP_CASE=<name> internal: run one case, print its resident bytes
 *      DMN_FP_ONLY=<sub>  run only cases whose group/name contains <sub>
 *      DMN_FP_STRICT=1    also fail on warnings (UNSHARED)
 *      DMN_FP_REPORT=1    never fail; print the table and exit 0
 *
 * Prints "SHFP: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <functional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <sys/wait.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <mach/mach_host.h>
#include <map>
#include <sys/mman.h>
#include <unistd.h>

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "dmn_d3d12_up.h" /* CreateCommittedResource1/2/3, CreateHeap1, ... */
#include "dmn_share.h"    /* dmn_share_anon_file */

#define T_TAG "SHFP"
#include "common/check.h"
#include "common/com.h"
#include "common/dx11.h"
#include "common/dx12.h"
#include "common/gpu.h"
#include "common/skip.h"
#include "common/util.h"

extern "C" unsigned long long dmn_test_metal_allocated_size(void);
extern "C" const char* dmn_test_metal_device_name(void);

namespace {

/* == Tunables ============================================================= */

uint64_t g_payload = 64ull << 20;
int g_n = 16;
bool g_strict = false;  /* also fail on warnings (UNSHARED) */
bool g_report = false;  /* never fail; print the table and exit 0 */
std::string g_only;

/* The gate, applied to the SLOPE (bytes reserved per byte of resource). Fixed
 * overheads are differenced out before this is applied, so the band left to
 * cover is only measurement noise — but the pool's 256 MiB quantization lands
 * unevenly across the two passes, so keep some room. 1.75 still separates a
 * clean 1.0 from a clean 2.0. */
double g_max_ratio = 1.75;
constexpr double kHigh = 1.35;

/* Resident bytes are noisier than reserved ones (other processes are in the
 * host-wide counter), so the resident gate is looser. A true 2x still clears
 * it by a wide margin. */
double g_max_resident = 1.6;

constexpr uint32_t kTexW = 4096; /* BGRA8 -> 16 KiB rows, so a linear
                                    substitution has no stride padding to
                                    confuse the ratio with */
uint32_t tex_h() { return (uint32_t)(g_payload / (4ull * kTexW)); }

double mib(int64_t bytes) { return (double)bytes / (1024.0 * 1024.0); }
uint64_t metal_alloc() { return dmn_test_metal_allocated_size(); }

/* == What the machine thinks is in use ==================================== */
/* Host-wide (see the header): other processes are in the number too, which is
 * tolerable because the payloads are ~1 GiB and the deltas are differenced
 * across two passes, but it is why the resident thresholds are looser than
 * the reserved ones and why a case reports both. */
/* Touching is done in the first pass only: the second exists to difference out
 * fixed overheads from the RESERVED number, and leaving its resources untouched
 * keeps a second payload of pages off the machine. */
bool g_do_touch = true;
/* Set by the touch helpers when a case WOULD have written its resources but
 * touching was off. Lets the parent skip spawning a residency child for the
 * cases that have nothing to write — half of them — instead of paying a
 * process launch and two device creations to learn that. */
bool g_would_touch = false;

uint64_t host_used() {
    vm_statistics64_data_t v;
    mach_msg_type_number_t c = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v,
                          &c) != KERN_SUCCESS)
        return 0;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS)
        page = 16384;
    return (uint64_t)(v.wire_count + v.active_count + v.compressor_page_count) *
           (uint64_t)page;
}

/* == Memory-pressure guard =================================================
 * The whole point of this test is to hold gigabytes of resources; on a
 * framework that pools every resource in chunks it makes resident and never
 * gives back (GPTk <= 3.0), the PARENT process balloons across cases and can
 * exhaust the machine. A test must never take the machine down: when host
 * memory in use crosses this fraction of RAM, remaining cases are skipped
 * with a note rather than measured. */
constexpr double kMaxHostUsedFraction = 0.60;

uint64_t host_ram() {
    uint64_t mem = 0;
    size_t n = sizeof mem;
    if (sysctlbyname("hw.memsize", &mem, &n, nullptr, 0) != 0)
        return 0;
    return mem;
}

bool memory_pressure() {
    const uint64_t ram = host_ram();
    if (!ram)
        return false;
    return (double)host_used() > kMaxHostUsedFraction * (double)ram;
}

/* == Telling a second allocation from a second mapping ==================== */
/*
 * currentAllocatedSize counts a substituted backing once per Metal object
 * wrapping it, and a same-process open of a shared handle reuses the
 * producer's MTLBuffer outright (dmn_share_metal.mm's alias cache), so the
 * `OpenSharedHandle xN` rows legitimately read a reserved slope of 0 or below
 * once the declared re-wrap is subtracted. That is aliasing working, not a
 * doubling, and harmless to the verdict (reserved never gates on its own).
 * Only a genuinely separate allocation costs the memory twice.
 *
 * The two are told apart with two independent facts:
 *
 *  - Each case DECLARES how many bytes of its Metal total are re-wraps of an
 *    already-counted object, because the case built the scenario and knows: N
 *    producers opened k times each is N*k*payload of re-wrap. Subtracting that
 *    leaves the distinct backing, which is what the verdict judges.
 *  - The shm census below independently COUNTS the distinct objects, which is
 *    what makes the declaration honest rather than assumed. An import that
 *    quietly copied instead of aliasing would produce 2N objects where the case
 *    says N, and is reported as COPIES. A shadow that is a Metal allocation
 *    rather than an shm object shows up in the same check from the other side:
 *    the Metal total exceeds the declared re-wraps plus the census.
 *
 * Identity is pshm_name, which outlives the immediate shm_unlink and is shared
 * by dups; fds are NOT a proxy for mappings (each registration dups its own,
 * so a producer holds two fds for one object). Names are 8 hex digits of
 * arc4random, so two live objects colliding is a ~2^-32 event per create.
 */
struct ShmCensus {
    uint64_t object_bytes = 0; /* summed over DISTINCT objects */
    int fds = 0, objects = 0;
};

ShmCensus shm_census() {
    ShmCensus c;
    std::map<std::string, uint64_t> objs;
    for (int fd = 0; fd < 4096; fd++) {
        if (fcntl(fd, F_GETFD) == -1)
            continue;
        struct pshm_fdinfo pi;
        const int n = proc_pidfdinfo(getpid(), fd, PROC_PIDFDPSHMINFO, &pi,
                                     sizeof pi);
        if (n < (int)sizeof pi)
            continue; /* not a POSIX shm fd */
        objs[std::string(pi.pshminfo.pshm_name)] =
            (uint64_t)pi.pshminfo.pshm_stat.vst_size;
        c.fds++;
    }
    for (const auto& kv : objs)
        c.object_bytes += kv.second;
    c.objects = (int)objs.size();
    return c;
}

/* == Case harness ========================================================= */

/* A case creates its resources, declares how many bytes of backing SHOULD
 * exist for them, calls mark() with everything still held, and returns. The
 * harness reads the allocator before it runs and again after its holders have
 * gone out of scope. */
struct Probe {
    uint64_t base = 0;
    uint64_t base_host = 0;
    ShmCensus base_shm;
    uint64_t expected = 0;
    uint64_t rewrap = 0;  /* declared bytes that re-wrap an object already counted */
    bool wants_shm = false; /* the case asked for sharing, so a substituted
                               shm backing must exist */
    int64_t live = 0;     /* Metal bytes, every wrapper counted */
    int64_t distinct = 0; /* the same with declared re-wraps removed */
    int64_t resident = 0; /* host-wide pages held, once everything is touched */
    bool touched = false; /* the case GPU-touched what it made, so `resident`
                             is the real cost rather than "nothing read it" */
    int64_t shm_obj = 0;  /* census: bytes in distinct shm objects */
    int shm_objs = 0;
    std::string note;

    /* `alias_bytes`: of `bytes`, how much is a second (third, ...) Metal object
     * over shm this case already accounted for. 0 for a plain producer. */
    void expect(uint64_t bytes, uint64_t alias_bytes = 0, bool shared = false) {
        expected = bytes;
        rewrap = alias_bytes;
        wants_shm = shared;
    }
    /* Call after the resources have been written by the GPU. The page
     * accounting lags GPU completion (see the header), so wait for the number
     * to stop climbing and take the plateau. */
    void mark_resident() {
        int64_t best = 0;
        int stable = 0;
        for (unsigned waited = 0; waited < 8000 && stable < 3; waited += 250) {
            sleep_ms(250);
            const int64_t now = (int64_t)host_used() - (int64_t)base_host;
            if (now > best + (8 << 20)) {
                best = now;
                stable = 0;
            } else {
                if (now > best)
                    best = now;
                stable++;
            }
        }
        resident = best;
        touched = true;
    }
    void mark() {
        live = (int64_t)metal_alloc() - (int64_t)base;
        if (!touched)
            resident = (int64_t)host_used() - (int64_t)base_host;
        distinct = live - (int64_t)rewrap;
        const ShmCensus now = shm_census();
        shm_obj = (int64_t)now.object_bytes - (int64_t)base_shm.object_bytes;
        shm_objs = now.objects - base_shm.objects;
    }
    void say(const std::string& s) { note = s; }
};

struct Row {
    std::string group, name, note;
    std::string control; /* name of the unshared control this row is judged
                            against, "" if none */
    uint64_t expected = 0;
    int64_t live = 0, distinct = 0, shm_obj = 0, resid = 0;
    int64_t resident = 0;
    bool touched = false;
    int shm_objs = 0, d_fds = 0;
    double slope = 0;   /* reserved multiplier: d(distinct)/d(payload), N fixed */
    double rslope = 0;  /* the same for RESIDENT bytes — what the verdict uses */
    double fixed = 0; /* the intercept: per-resource + per-case overhead */
    bool ran = false;
    bool marked = false;
    bool wants_shm = false;
};

std::string hr_str(const char* what, HRESULT hr) {
    char b[96];
    snprintf(b, sizeof b, "%s returned 0x%08x", what, (unsigned)hr);
    return b;
}

std::vector<Row> g_rows;

using CaseFn = std::function<bool(Probe&)>;

struct CaseDef {
    std::string group, name;
    CaseFn fn;
    std::string control; /* see Row::control */
};
std::vector<CaseDef> g_cases;

/* Wait for host-wide memory to stop moving. Releases land asynchronously, and
 * an unsettled baseline is what turns a 1x case into a negative reading. */
void settle_host(unsigned max_ms = 4000) {
    uint64_t prev = host_used();
    unsigned stable = 0;
    for (unsigned waited = 0; waited < max_ms && stable < 3; waited += 200) {
        sleep_ms(200);
        const uint64_t now = host_used();
        const int64_t d = (int64_t)now - (int64_t)prev;
        stable = (d > -(32 << 20) && d < (32 << 20)) ? stable + 1 : 0;
        prev = now;
    }
}

/* One measurement of a case at the payload currently in g_payload. Measures
 * RESERVED bytes only, which come from Metal's allocator and are immediate —
 * no host-memory settling needed, and skipping it keeps the parent's two
 * passes cheap. Residency settles in the child, where it matters. */
bool run_pass(const CaseFn& fn, Probe* out) {
    g_would_touch = false;
    Probe p;
    p.base_shm = shm_census();
    p.base_host = host_used();
    p.base = metal_alloc();
    const bool ran = fn(p);
    /* Deferred destruction: both runtimes retire on GPU completion. */
    sleep_ms(250);
    *out = p;
    return ran;
}

/* Each case is measured TWICE, at payload P and 2P with N held fixed, and the
 * multiplier is the SLOPE between them:
 *
 *     distinct(T) = k*T + c*N + F,  T = N*payload
 *     k = (distinct(2P) - distinct(P)) / (N*P)
 *
 * because a single ratio distinct/(N*payload) cannot tell a multiplier from an
 * overhead: it reports k + (c*N + F)/(N*payload) and blames all of it on k.
 * Differencing two payloads at fixed N cancels BOTH constant terms — the
 * per-resource overhead c (a header, a keyed-mutex page) and the per-case
 * overhead F (most importantly D3DMetal's 256 MiB pool quantization, which is
 * what made the unshared controls read 1.5x-2.0x and vary with N).
 *
 * So k is the honest "does this allocate the payload twice" number, and the
 * intercept is reported beside it as the fixed cost it actually is. */
/* Registration only — measurement happens later, because the resident half of
 * each case runs in a child process and the parent needs the list first. */
/* `control`: the unshared case that shows what the framework charges for this
 * resource kind on its own. Where that control itself reads as `pool` (the
 * framework parks EVERY resource of the kind in a pool slot it makes resident
 * — GPTk 3.0 and earlier, which have no dedicated-allocation knob), a shared
 * row cannot be judged absolutely: our shm is one copy on top of a slot the
 * substitution has no way to give back, and every extra wrapper (an open, an
 * alias) buys another slot. Such rows are REPORTED as `pooled`, not gated.
 * Where the control reads ~1x (4.0: dedicated allocations), the shared row is
 * judged strictly, which is the case that matters. A control that is
 * `untouched` says nothing and leaves the row strict. */
void run_case(const char* group, const char* name, const CaseFn& fn,
              const char* control = "") {
    if (!g_only.empty() &&
        std::string(group).find(g_only) == std::string::npos &&
        std::string(name).find(g_only) == std::string::npos)
        return;
    g_cases.push_back(CaseDef{group, name, fn, control});
}

/* == Resident bytes, measured in a child process ========================== */
/* Each case's resident measurement runs in a fresh child (see the header for
 * why): it re-execs this binary with DMN_FP_CASE set, the child runs that one
 * case, writes what the machine's page count did, and dies. */
constexpr const char* kResidentPrefix = "RESIDENT_BYTES ";

bool measure_resident_child(const CaseDef& c, int64_t* out) {
    char exe[4096];
    uint32_t n = sizeof exe;
    if (_NSGetExecutablePath(exe, &n) != 0)
        return false;

    int fds[2];
    if (pipe(fds) != 0)
        return false;

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&fa, fds[0]);

    char case_env[512], mb_env[64], n_env[64];
    snprintf(case_env, sizeof case_env, "DMN_FP_CASE=%s", c.name.c_str());
    snprintf(mb_env, sizeof mb_env, "DMN_FP_MB=%llu",
             (unsigned long long)(g_payload >> 20));
    snprintf(n_env, sizeof n_env, "DMN_FP_N=%d", g_n);
    char log_env[] = "DMN_LOG=error";
    const char* fw = getenv("D3DMETAL_FRAMEWORK_PATH");
    std::string fw_env = std::string("D3DMETAL_FRAMEWORK_PATH=") + (fw ? fw : "");
    char* envp[] = {case_env, mb_env, n_env, log_env,
                    fw ? (char*)fw_env.c_str() : nullptr, nullptr};
    char* argv[] = {exe, nullptr};

    pid_t pid = 0;
    const int rc = posix_spawn(&pid, exe, &fa, nullptr, argv, envp);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) {
        close(fds[0]);
        return false;
    }

    std::string buf;
    char tmp[512];
    ssize_t got;
    while ((got = read(fds[0], tmp, sizeof tmp)) > 0)
        buf.append(tmp, (size_t)got);
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);

    const size_t at = buf.find(kResidentPrefix);
    if (at == std::string::npos)
        return false;
    return sscanf(buf.c_str() + at + strlen(kResidentPrefix), "%lld",
                  (long long*)out) == 1;
}

void measure_case(const CaseDef& cd) {
    const char* group = cd.group.c_str();
    const char* name = cd.name.c_str();
    const CaseFn& fn = cd.fn;

    Row row;
    row.group = group;
    row.name = name;
    if (memory_pressure()) {
        row.note = "skipped: host memory pressure (framework retains pooled "
                   "chunks across cases)";
        row.control = cd.control;
        printf(T_TAG ": %-8s %-34s SKIPPED — host memory in use above %.0f%% "
               "of RAM\n", group, name, kMaxHostUsedFraction * 100.0);
        fflush(stdout);
        g_rows.push_back(row);
        return;
    }

    const uint64_t p1 = g_payload, p2 = g_payload * 2;
    const int base_fds = t_count_fds();

    Probe a, b;
    g_do_touch = false; /* the parent measures RESERVED bytes only */
    g_payload = p1;
    const bool ran_a = run_pass(fn, &a);
    g_payload = p2;
    const bool ran_b = run_pass(fn, &b);
    g_payload = p1;

    const bool touchable = g_would_touch;
    row.ran = ran_a && ran_b;
    row.d_fds = t_count_fds() - base_fds;
    /* Everything except the slope is reported from the small pass. */
    row.expected = a.expected;
    row.live = a.live;
    row.distinct = a.distinct;
    row.shm_obj = a.shm_obj;
    row.shm_objs = a.shm_objs;
    /* Resident bytes come from a fresh process, for the reasons above — but
     * only for cases that write something, since an untouched case's resident
     * figure is uninformative and the child is expensive. */
    int64_t child_resident = 0;
    row.touched = row.ran && touchable &&
                  measure_resident_child(cd, &child_resident);
    row.resident = child_resident;
    row.wants_shm = a.wants_shm;
    row.control = cd.control;
    row.marked = a.expected != 0 && b.expected != 0;
    row.note = a.note.empty() ? b.note : a.note;

    if (row.marked && b.expected > a.expected) {
        /* Difference against what the CASE declared it needs, not against the
         * raw payload: a case whose expectation is not simply N*payload (a
         * texture heap is sized to fit alignment, so it is 2N*payload) would
         * otherwise read a multiple of its own scaling factor. */
        const double dT = (double)(b.expected - a.expected);
        row.slope = (double)(b.distinct - a.distinct) / dT;
        row.fixed = (double)a.distinct - row.slope * (double)a.expected;
        row.rslope = (double)row.resident / (double)a.expected;
    }

    /* A framework that pools D3D12 buffers/heaps in chunks it makes resident
     * and never gives back retains every case's footprint across cases; scale
     * N down for the rest of the run so the machine survives (each case's
     * expectation is computed from g_n at run time, so rows stay
     * self-consistent). Keyed on a D3D12 BUFFER/HEAP control's RESERVED slope:
     * chunk quantization pushes it to 2x-4x on such a framework and leaves it
     * at 1.0 on one that allocates dedicated, whereas the resident reading sits
     * near the threshold and would flip run to run; and 4.0 pools textures but
     * not D3D12 buffers, so a texture control must not trigger it. */
    static bool pool_scaled = false;
    if (!pool_scaled && !row.wants_shm && row.marked && row.slope >= 2.0 &&
        g_n > 4 && !strcmp(group, "d3d12") &&
        (strstr(name, "buf") || strstr(name, "CreateHeap only"))) {
        pool_scaled = true;
        g_n = 4;
        printf(T_TAG ": control %s reserves %.2fx per byte — framework pools "
               "every resource; scaling N to %d for the remaining cases\n",
               name, row.slope, g_n);
    }
    printf(T_TAG ": %-8s %-34s reserved %5.2fx  resident %5.2fx%s%s\n",
           group, name, row.slope, row.rslope,
           row.touched ? "" : " (no resident measurement)",
           row.ran ? (row.marked ? "" : "  [no measurement]") : "  [SKIPPED]");
    fflush(stdout);
    g_rows.push_back(row);
}

/* == Shared state ========================================================= */

struct Env {
    Com<ID3D11Device> d11;
    Com<ID3D11DeviceContext> ctx11;
    Com<ID3D11Device1> d11_1;
    Com<ID3D11Device3> d11_3;
    Com<ID3D11Device5> d11_5;
    Com<ID3D12Device> d12;
    Com<ID3D12CommandQueue> q12;
    Com<ID3D12CommandAllocator> alloc12;
    Com<ID3D12GraphicsCommandList> list12;
    Com<ID3D12Fence> fence12;
    UINT64 fv = 0;
    Com<ID3D12Resource> src12; /* copy source, sized for the LARGER pass and
                                  committed during setup so it is inside every
                                  baseline rather than inside a measurement */
};
Env g;

/* Submit `body` and wait for it to retire. */
bool gpu12_run(const std::function<void(ID3D12GraphicsCommandList*)>& body) {
    if (!g.q12 || FAILED(g.alloc12->Reset()) ||
        FAILED(g.list12->Reset(g.alloc12.ptr(), nullptr)))
        return false;
    body(g.list12.ptr());
    if (FAILED(g.list12->Close()))
        return false;
    ID3D12CommandList* lists[] = {g.list12.ptr()};
    g.q12->ExecuteCommandLists(1, lists);
    const UINT64 want = ++g.fv;
    if (FAILED(g.q12->Signal(g.fence12.ptr(), want)))
        return false;
    const uint64_t deadline = now_ms() + 20000;
    while (g.fence12->GetCompletedValue() < want) {
        if (now_ms() > deadline) {
            fprintf(stderr, T_TAG ": D3D12 queue did not retire within 20s\n");
            return false;
        }
        sleep_ms(1);
    }
    return true;
}

/* Make every byte of these buffers real. A GPU write commits a whole Metal
 * allocation at once, so this is what turns a reservation into resident pages
 * — and what proves a reservation nothing writes stays virtual. */
bool touch12_bufs(const std::vector<Com<ID3D12Resource>>& v, uint64_t bytes) {
    if (v.empty() || !g.src12)
        return false;
    if (!g_do_touch) {
        g_would_touch = true;
        return false;
    }
    return gpu12_run([&](ID3D12GraphicsCommandList* cl) {
        for (const auto& r : v)
            if (r)
                cl->CopyBufferRegion(r.ptr(), 0, g.src12.ptr(), 0, bytes);
    });
}

bool touch11_texs(const std::vector<Com<ID3D11Texture2D>>& v) {
    if (v.empty())
        return false;
    if (!g_do_touch) {
        g_would_touch = true;
        return false;
    }
    const float c[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    for (const auto& t : v) {
        Com<ID3D11RenderTargetView> rtv;
        if (!t || FAILED(g.d11->CreateRenderTargetView(t.ptr(), nullptr, &rtv)) ||
            !rtv)
            return false;
        g.ctx11->ClearRenderTargetView(rtv.ptr(), c);
    }
    g.ctx11->Flush();
    return t_gpu_queue_alive_d3d11(g.d11.ptr(), g.ctx11.ptr(), 20000);
}

D3D11_TEXTURE2D_DESC t2d(UINT misc) {
    D3D11_TEXTURE2D_DESC td{};
    td.Width = kTexW;
    td.Height = tex_h();
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = misc;
    return td;
}

D3D11_BUFFER_DESC bd11(UINT misc) {
    D3D11_BUFFER_DESC b{};
    b.ByteWidth = (UINT)g_payload;
    b.Usage = D3D11_USAGE_DEFAULT;
    b.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    b.MiscFlags = misc;
    return b;
}

D3D12_RESOURCE_DESC buf_desc(uint64_t bytes) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return rd;
}

D3D12_RESOURCE_DESC tex_desc() {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = kTexW;
    rd.Height = tex_h();
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    return rd;
}

/* The ...1/2/3 entry points take the SamplerFeedback-extended desc; the extra
 * field is zero for everything here. D3D12_BARRIER_LAYOUT_COMMON is 0 (the
 * enum itself is only forwarded, so dmn_d3d12_up.h types it as UINT). */
D3D12_RESOURCE_DESC1 desc1_of(const D3D12_RESOURCE_DESC& rd) {
    D3D12_RESOURCE_DESC1 d{};
    d.Dimension = rd.Dimension;
    d.Alignment = rd.Alignment;
    d.Width = rd.Width;
    d.Height = rd.Height;
    d.DepthOrArraySize = rd.DepthOrArraySize;
    d.MipLevels = rd.MipLevels;
    d.Format = rd.Format;
    d.SampleDesc = rd.SampleDesc;
    d.Layout = rd.Layout;
    d.Flags = rd.Flags;
    return d;
}
constexpr DMN_D3D12_BARRIER_LAYOUT kLayoutCommon = 0;

HRESULT committed(D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC& rd,
                  D3D12_HEAP_FLAGS flags, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type;
    return g.d12->CreateCommittedResource(&hp, flags, &rd,
                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                          __uuidof(ID3D12Resource), (void**)out);
}

/* A create that a given D3DMetal simply does not implement is a gap in the
 * framework, not a finding — say so and drop the row. */
bool unimpl(Probe& p, HRESULT hr) {
    if (t_unimplemented(hr)) {
        p.say("not implemented by this D3DMetal");
        return true;
    }
    return false;
}

/* == D3D11 producers ====================================================== */

bool c11_tex(Probe& p, UINT misc) {
    std::vector<Com<ID3D11Texture2D>> v(g_n);
    D3D11_TEXTURE2D_DESC td = t2d(misc);
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = g.d11->CreateTexture2D(&td, nullptr, &v[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !v[i])
            return false;
    }
    p.expect((uint64_t)g_n * g_payload, 0, misc != 0);
    p.mark();
    if (touch11_texs(v))
        p.mark_resident();
    return true;
}

bool c11_tex1(Probe& p) {
    if (!g.d11_3) {
        p.say("ID3D11Device3 unavailable");
        return false;
    }
    D3D11_TEXTURE2D_DESC b = t2d(D3D11_RESOURCE_MISC_SHARED);
    D3D11_TEXTURE2D_DESC1 td{};
    memcpy(&td, &b, sizeof(b));
    td.TextureLayout = D3D11_TEXTURE_LAYOUT_UNDEFINED;
    std::vector<Com<ID3D11Texture2D1>> v(g_n);
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = g.d11_3->CreateTexture2D1(&td, nullptr, &v[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !v[i])
            return false;
    }
    p.expect((uint64_t)g_n * g_payload, 0, true);
    p.mark();
    return true;
}

bool c11_buf(Probe& p, UINT misc) {
    std::vector<Com<ID3D11Buffer>> v(g_n);
    D3D11_BUFFER_DESC bd = bd11(misc);
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = g.d11->CreateBuffer(&bd, nullptr, &v[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !v[i])
            return false;
    }
    p.expect((uint64_t)g_n * g_payload, 0, misc != 0);
    p.mark();
    return true;
}

bool c11_tex3d(Probe& p, UINT misc) {
    D3D11_TEXTURE3D_DESC td{};
    td.Width = 256;
    td.Height = 256;
    td.Depth = (UINT)(g_payload / (256 * 256 * 4));
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = misc;
    std::vector<Com<ID3D11Texture3D>> v(g_n);
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = g.d11->CreateTexture3D(&td, nullptr, &v[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !v[i]) {
            p.say(misc ? "shared 3D texture create refused" : "3D create refused");
            return false;
        }
    }
    p.expect((uint64_t)g_n * g_payload, 0, misc != 0);
    p.mark();
    return true;
}

/* == D3D11 export ========================================================= */
/* An export must cost an fd and a POD, never a copy of the payload — including
 * repeated exports of one resource, where a per-handle shadow would show up as
 * per_res x payload. Expected backing is therefore N payloads however many
 * handles are taken. */

bool c11_export(Probe& p, bool nt, int per_res) {
    std::vector<Com<ID3D11Texture2D>> v(g_n);
    std::vector<HANDLE> handles;
    D3D11_TEXTURE2D_DESC td = t2d(D3D11_RESOURCE_MISC_SHARED);
    for (int i = 0; i < g_n; i++) {
        if (FAILED(g.d11->CreateTexture2D(&td, nullptr, &v[i])) || !v[i])
            return false;
        for (int k = 0; k < per_res; k++) {
            HANDLE h = nullptr;
            if (nt) {
                Com<IDXGIResource1> r1;
                HRESULT hr = v[i]->QueryInterface(__uuidof(IDXGIResource1),
                                                  (void**)&r1);
                if (FAILED(hr) || !r1) {
                    p.say(hr_str("QueryInterface(IDXGIResource1)", hr));
                    return false;
                }
                hr = r1->CreateSharedHandle(
                    nullptr,
                    DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                    nullptr, &h);
                if (unimpl(p, hr))
                    return false;
                if (FAILED(hr) || !h) {
                    p.say(hr_str("IDXGIResource1::CreateSharedHandle", hr));
                    return false;
                }
            } else {
                Com<IDXGIResource> r;
                if (FAILED(v[i]->QueryInterface(__uuidof(IDXGIResource),
                                                (void**)&r)) ||
                    FAILED(r->GetSharedHandle(&h)))
                    return false;
            }
            handles.push_back(h);
        }
    }
    p.expect((uint64_t)g_n * g_payload, 0, true);
    p.mark();
    if (nt)
        for (HANDLE h : handles)
            dmn_shared_handle_close(h);
    return true;
}

/* == D3D11 import ========================================================= */
/* An open must ALIAS the producer's pages. N producers each opened `opens`
 * times still needs exactly N payloads; a copying import would report
 * N * (1 + opens). */

bool c11_open(Probe& p, int opens, bool via_1) {
    if (via_1 && !g.d11_1) {
        p.say("ID3D11Device1 unavailable");
        return false;
    }
    std::vector<Com<ID3D11Texture2D>> prod(g_n);
    std::vector<Com<ID3D11Texture2D>> opened;
    D3D11_TEXTURE2D_DESC td = t2d(D3D11_RESOURCE_MISC_SHARED);
    for (int i = 0; i < g_n; i++) {
        if (FAILED(g.d11->CreateTexture2D(&td, nullptr, &prod[i])) || !prod[i])
            return false;
        Com<IDXGIResource> r;
        HANDLE h = nullptr;
        if (FAILED(prod[i]->QueryInterface(__uuidof(IDXGIResource), (void**)&r)) ||
            FAILED(r->GetSharedHandle(&h)))
            return false;
        for (int k = 0; k < opens; k++) {
            Com<ID3D11Texture2D> o;
            HRESULT hr = via_1
                ? g.d11_1->OpenSharedResource1(h, __uuidof(ID3D11Texture2D),
                                               (void**)&o)
                : g.d11->OpenSharedResource(h, __uuidof(ID3D11Texture2D),
                                            (void**)&o);
            if (unimpl(p, hr))
                return false;
            if (FAILED(hr) || !o)
                return false;
            opened.push_back(o);
        }
    }
    /* Each open re-wraps the producer's object; only the producers are new. */
    p.expect((uint64_t)g_n * g_payload, (uint64_t)g_n * opens * g_payload, true);
    p.mark();
    return true;
}

/* == D3D12 producers ====================================================== */

bool c12_committed(Probe& p, D3D12_HEAP_FLAGS flags, bool texture) {
    std::vector<Com<ID3D12Resource>> v(g_n);
    D3D12_RESOURCE_DESC rd = texture ? tex_desc() : buf_desc(g_payload);
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = committed(D3D12_HEAP_TYPE_DEFAULT, rd, flags, &v[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !v[i])
            return false;
    }
    p.expect((uint64_t)g_n * g_payload, 0, flags != D3D12_HEAP_FLAG_NONE);
    p.mark();
    if (!texture && touch12_bufs(v, g_payload))
        p.mark_resident();
    return true;
}

/* CreateCommittedResource1/2/3 — same shared create through the up-level
 * device interfaces, which are separate hooks and could diverge. */
bool c12_committed_v(Probe& p, int version, D3D12_HEAP_FLAGS flags) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = tex_desc();
    const D3D12_RESOURCE_DESC1 rd1 = desc1_of(rd);
    std::vector<Com<ID3D12Resource>> v(g_n);

    Com<ID3D12Device4> d4;
    Com<ID3D12Device8> d8;
    Com<ID3D12Device10> d10;
    if (version >= 1 && FAILED(g.d12->QueryInterface(__uuidof(ID3D12Device4),
                                                     (void**)&d4))) {
        p.say("ID3D12Device4 unavailable");
        return false;
    }
    if (version >= 2 && FAILED(g.d12->QueryInterface(__uuidof(ID3D12Device8),
                                                     (void**)&d8))) {
        p.say("ID3D12Device8 unavailable");
        return false;
    }
    if (version >= 3 && FAILED(g.d12->QueryInterface(__uuidof(ID3D12Device10),
                                                     (void**)&d10))) {
        p.say("ID3D12Device10 unavailable");
        return false;
    }

    for (int i = 0; i < g_n; i++) {
        HRESULT hr = E_NOTIMPL;
        if (version == 1)
            hr = d4->CreateCommittedResource1(&hp, flags, &rd,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, nullptr,
                                              __uuidof(ID3D12Resource),
                                              (void**)&v[i]);
        else if (version == 2)
            hr = d8->CreateCommittedResource2(&hp, flags, &rd1,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, nullptr,
                                              __uuidof(ID3D12Resource),
                                              (void**)&v[i]);
        else
            hr = d10->CreateCommittedResource3(&hp, flags, &rd1, kLayoutCommon,
                                               nullptr, nullptr, 0, nullptr,
                                               __uuidof(ID3D12Resource),
                                               (void**)&v[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !v[i])
            return false;
    }
    p.expect((uint64_t)g_n * g_payload, 0, flags != D3D12_HEAP_FLAG_NONE);
    p.mark();
    return true;
}

/* The prime suspect. A placed resource's memory is supposed to come OUT of the
 * heap the app already paid for, so N heaps each holding one placed resource
 * must cost N payloads whether or not the heap is shared. If the shared path
 * substitutes a fresh shm-backed buffer instead of using the heap's own bytes,
 * both are live and the app pays twice for one resource. */
bool c12_heap_placed(Probe& p, D3D12_HEAP_FLAGS flags, bool texture,
                     int heap_version, int placed_version) {
    std::vector<Com<ID3D12Heap>> heaps(g_n);
    std::vector<Com<ID3D12Resource>> res(g_n);

    Com<ID3D12Device4> d4;
    Com<ID3D12Device8> d8;
    Com<ID3D12Device10> d10;
    if ((heap_version >= 1 || placed_version >= 0) &&
        FAILED(g.d12->QueryInterface(__uuidof(ID3D12Device4), (void**)&d4)))
        d4 = nullptr;
    if (heap_version >= 1 && !d4) {
        p.say("ID3D12Device4 unavailable");
        return false;
    }
    if (placed_version >= 1 &&
        FAILED(g.d12->QueryInterface(__uuidof(ID3D12Device8), (void**)&d8))) {
        p.say("ID3D12Device8 unavailable");
        return false;
    }
    if (placed_version >= 2 &&
        FAILED(g.d12->QueryInterface(__uuidof(ID3D12Device10), (void**)&d10))) {
        p.say("ID3D12Device10 unavailable");
        return false;
    }

    /* A texture needs MORE heap than its logical bytes — tiling and placement
     * alignment push a 64 MiB render target over a 64 MiB heap, which is
     * E_INVALIDARG and reads exactly like "the framework refuses placed
     * textures". It does not: the same texture places fine in a 128 MiB heap.
     * Give textures room, and measure against the heap the app asked for
     * rather than the resource's logical size, so a correct implementation
     * still reads 1.0 either way. */
    const uint64_t heap_size = texture ? g_payload * 2 : g_payload;
    D3D12_HEAP_DESC hd{};
    hd.SizeInBytes = heap_size;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    /* ALLOW_ALL_BUFFERS_AND_TEXTURES (0) for textures: the ALLOW_ONLY_* tiers
     * are what a heap-tier-1 device needs, and restricting the heap is a second
     * reason a placed create can be refused — not what this is measuring. */
    hd.Flags = flags | (texture ? D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES
                                : D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
    D3D12_RESOURCE_DESC rd = texture ? tex_desc() : buf_desc(g_payload);
    const D3D12_RESOURCE_DESC1 rd1 = desc1_of(rd);

    for (int i = 0; i < g_n; i++) {
        HRESULT hr = heap_version >= 1
            ? d4->CreateHeap1(&hd, nullptr, __uuidof(ID3D12Heap), (void**)&heaps[i])
            : g.d12->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&heaps[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !heaps[i])
            return false;

        if (placed_version == 1)
            hr = d8->CreatePlacedResource1(heaps[i].ptr(), 0, &rd1,
                                           D3D12_RESOURCE_STATE_COMMON, nullptr,
                                           __uuidof(ID3D12Resource),
                                           (void**)&res[i]);
        else if (placed_version == 2)
            hr = d10->CreatePlacedResource2(heaps[i].ptr(), 0, &rd1,
                                            kLayoutCommon, nullptr,
                                            0, nullptr, __uuidof(ID3D12Resource),
                                            (void**)&res[i]);
        else
            hr = g.d12->CreatePlacedResource(heaps[i].ptr(), 0, &rd,
                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             __uuidof(ID3D12Resource),
                                             (void**)&res[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !res[i]) {
            p.say(hr_str(texture ? "CreatePlacedResource(texture)"
                                 : "CreatePlacedResource(buffer)", hr));
            return false;
        }
    }
    p.expect((uint64_t)g_n * heap_size, 0, flags != D3D12_HEAP_FLAG_NONE);
    p.mark();
    if (!texture && touch12_bufs(res, g_payload))
        p.mark_resident();
    return true;
}

/* D3D12's whole point for placed resources: two resources placed at the same
 * heap offset ALIAS — same bytes, and the app aliases them deliberately to
 * reuse memory. `dup` buffers are placed at offset 0 of each heap, so a correct
 * implementation shows N shm objects (one per heap) no matter how large `dup`
 * is. A fresh backing per placed create shows N*dup, which is unbounded
 * RESIDENT memory as soon as the app writes them, and is what the COPIES
 * verdict flags. */
bool c12_heap_alias(Probe& p, int dup) {
    std::vector<Com<ID3D12Heap>> heaps(g_n);
    std::vector<Com<ID3D12Resource>> res;
    D3D12_HEAP_DESC hd{};
    hd.SizeInBytes = g_payload;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hd.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    D3D12_RESOURCE_DESC rd = buf_desc(g_payload);
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = g.d12->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&heaps[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !heaps[i])
            return false;
        for (int k = 0; k < dup; k++) {
            Com<ID3D12Resource> r;
            hr = g.d12->CreatePlacedResource(heaps[i].ptr(), 0, &rd,
                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             __uuidof(ID3D12Resource), (void**)&r);
            if (FAILED(hr) || !r) {
                p.say(hr_str("CreatePlacedResource(alias)", hr));
                return false;
            }
            res.push_back(r);
        }
    }
    /* dup placements of one heap offset are the SAME bytes, so all but the
     * first are re-wraps — if the implementation aliases at all. */
    p.expect((uint64_t)g_n * g_payload,
             (uint64_t)g_n * (dup - 1) * g_payload, true);
    p.mark();
    if (touch12_bufs(res, g_payload))
        p.mark_resident();
    return true;
}

/* A shared heap with NOTHING placed in it: isolates the heap's own allocation
 * from the placed resource's, so the doubling can be attributed to one or the
 * other rather than to the pair. */
bool c12_heap_only(Probe& p, D3D12_HEAP_FLAGS flags) {
    std::vector<Com<ID3D12Heap>> heaps(g_n);
    D3D12_HEAP_DESC hd{};
    hd.SizeInBytes = g_payload;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hd.Flags = flags | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = g.d12->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&heaps[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !heaps[i])
            return false;
    }
    p.expect((uint64_t)g_n * g_payload);
    p.mark();
    return true;
}

/* == D3D12 export / import ================================================ */

bool c12_export(Probe& p, int per_res) {
    std::vector<Com<ID3D12Resource>> v(g_n);
    std::vector<HANDLE> handles;
    D3D12_RESOURCE_DESC rd = buf_desc(g_payload);
    for (int i = 0; i < g_n; i++) {
        if (FAILED(committed(D3D12_HEAP_TYPE_DEFAULT, rd,
                             D3D12_HEAP_FLAG_SHARED, &v[i])) || !v[i])
            return false;
        for (int k = 0; k < per_res; k++) {
            HANDLE h = nullptr;
            HRESULT hr = g.d12->CreateSharedHandle(v[i].ptr(), nullptr, 0,
                                                   nullptr, &h);
            if (unimpl(p, hr))
                return false;
            if (FAILED(hr))
                return false;
            handles.push_back(h);
        }
    }
    p.expect((uint64_t)g_n * g_payload, 0, true);
    p.mark();
    if (touch12_bufs(v, g_payload))
        p.mark_resident();
    for (HANDLE h : handles)
        dmn_shared_handle_close(h);
    return true;
}

bool c12_open(Probe& p, int opens) {
    std::vector<Com<ID3D12Resource>> prod(g_n), opened;
    std::vector<HANDLE> handles;
    D3D12_RESOURCE_DESC rd = buf_desc(g_payload);
    for (int i = 0; i < g_n; i++) {
        if (FAILED(committed(D3D12_HEAP_TYPE_DEFAULT, rd,
                             D3D12_HEAP_FLAG_SHARED, &prod[i])) || !prod[i])
            return false;
        HANDLE h = nullptr;
        if (FAILED(g.d12->CreateSharedHandle(prod[i].ptr(), nullptr, 0, nullptr,
                                             &h)))
            return false;
        handles.push_back(h);
        for (int k = 0; k < opens; k++) {
            Com<ID3D12Resource> o;
            HRESULT hr = g.d12->OpenSharedHandle(h, __uuidof(ID3D12Resource),
                                                 (void**)&o);
            if (unimpl(p, hr))
                return false;
            if (FAILED(hr) || !o)
                return false;
            opened.push_back(o);
        }
    }
    p.expect((uint64_t)g_n * g_payload, (uint64_t)g_n * opens * g_payload, true);
    p.mark();
    /* Touch through the OPENED views: if an import copied instead of aliasing,
     * the duplicate pages become resident here. */
    if (touch12_bufs(opened, g_payload))
        p.mark_resident();
    for (HANDLE h : handles)
        dmn_shared_handle_close(h);
    return true;
}

bool c_xapi_11_to_12(Probe& p) {
    std::vector<Com<ID3D11Texture2D>> prod(g_n);
    std::vector<Com<ID3D12Resource>> opened(g_n);
    D3D11_TEXTURE2D_DESC td = t2d(D3D11_RESOURCE_MISC_SHARED);
    for (int i = 0; i < g_n; i++) {
        if (FAILED(g.d11->CreateTexture2D(&td, nullptr, &prod[i])) || !prod[i])
            return false;
        Com<IDXGIResource> r;
        HANDLE h = nullptr;
        if (FAILED(prod[i]->QueryInterface(__uuidof(IDXGIResource), (void**)&r)) ||
            FAILED(r->GetSharedHandle(&h)))
            return false;
        HRESULT hr = g.d12->OpenSharedHandle(h, __uuidof(ID3D12Resource),
                                             (void**)&opened[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !opened[i]) {
            p.say("D3D12 could not open the D3D11 handle");
            return false;
        }
    }
    /* No residency touch: the opened resources are TEXTURES, and the copy
     * helper here writes buffers. */
    p.expect((uint64_t)g_n * g_payload, (uint64_t)g_n * g_payload, true);
    p.mark();
    return true;
}

bool c_xapi_12_to_11(Probe& p) {
    std::vector<Com<ID3D12Resource>> prod(g_n);
    std::vector<Com<ID3D11Texture2D>> opened(g_n);
    std::vector<HANDLE> handles;
    D3D12_RESOURCE_DESC rd = tex_desc();
    for (int i = 0; i < g_n; i++) {
        if (FAILED(committed(D3D12_HEAP_TYPE_DEFAULT, rd,
                             D3D12_HEAP_FLAG_SHARED, &prod[i])) || !prod[i])
            return false;
        HANDLE h = nullptr;
        if (FAILED(g.d12->CreateSharedHandle(prod[i].ptr(), nullptr, 0, nullptr,
                                             &h)))
            return false;
        handles.push_back(h);
        HRESULT hr = g.d11->OpenSharedResource(h, __uuidof(ID3D11Texture2D),
                                               (void**)&opened[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !opened[i]) {
            p.say("D3D11 could not open the D3D12 handle");
            return false;
        }
    }
    p.expect((uint64_t)g_n * g_payload, (uint64_t)g_n * g_payload, true);
    p.mark();
    for (HANDLE h : handles)
        dmn_shared_handle_close(h);
    return true;
}

/* The Neptune import path: the caller already owns the pages, so an imported
 * heap — with or without a buffer placed over it — should cost nothing beyond
 * the caller's own payload.
 *
 * `place` splits the two halves apart, the same way "CreateHeap only" does for
 * the native path: with place=false nothing is placed, so whatever the import
 * costs is the heap reservation alone. Neither half declares a re-wrap: the
 * placed buffer maps the CALLER's fd (counted once by the census), and any
 * heap reservation D3DMetal makes is separate memory of its own. */
bool c_heap_import(Probe& p, bool place) {
    std::vector<int> fds;
    std::vector<Com<ID3D12Heap>> heaps;
    std::vector<Com<ID3D12Resource>> res;
    const GUID iid_heap = __uuidof(ID3D12Heap);
    bool ok = true;

    for (int i = 0; i < g_n && ok; i++) {
        const int fd = dmn_share_anon_file((size_t)g_payload);
        if (fd < 0) {
            ok = false;
            break;
        }
        fds.push_back(fd);
        ID3D12Heap* raw = nullptr;
        HRESULT hr = (HRESULT)dmn_open_existing_heap_from_fd(
            g.d12.ptr(), fd, 0, g_payload, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_HEAP_FLAG_NONE, &iid_heap, (void**)&raw);
        if (FAILED(hr) || !raw) {
            p.say("dmn_open_existing_heap_from_fd failed");
            ok = false;
            break;
        }
        Com<ID3D12Heap> heap;
        heap = raw;
        raw->Release();
        heaps.push_back(heap);

        if (!place)
            continue;
        D3D12_RESOURCE_DESC rd = buf_desc(g_payload);
        Com<ID3D12Resource> r;
        hr = g.d12->CreatePlacedResource(heap.ptr(), 0, &rd,
                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                         __uuidof(ID3D12Resource), (void**)&r);
        if (FAILED(hr) || !r) {
            p.say(hr_str("CreatePlacedResource on the imported heap", hr));
            ok = false;
            break;
        }
        res.push_back(r);
    }
    if (ok) {
        p.expect((uint64_t)g_n * g_payload, 0, place);
        p.mark();
        if (place && touch12_bufs(res, g_payload))
            p.mark_resident();
    }
    res.clear();
    heaps.clear();
    for (int fd : fds)
        close(fd);
    return ok;
}

/* == Fences =============================================================== */
/* A fence's own payload is one page, so there is no ratio to take — these hold
 * N of them and report the absolute cost, which is where a per-fence shadow
 * buffer or a leaked helper allocation shows up. */

bool c12_fence(Probe& p, bool shared, bool export_handle) {
    std::vector<Com<ID3D12Fence>> v(g_n);
    std::vector<HANDLE> handles;
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = g.d12->CreateFence(0, shared ? D3D12_FENCE_FLAG_SHARED
                                                  : D3D12_FENCE_FLAG_NONE,
                                        __uuidof(ID3D12Fence), (void**)&v[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !v[i])
            return false;
        if (export_handle) {
            HANDLE h = nullptr;
            if (SUCCEEDED(g.d12->CreateSharedHandle(v[i].ptr(), nullptr, 0,
                                                    nullptr, &h)))
                handles.push_back(h);
        }
    }
    p.expect((uint64_t)g_n * 16384, 0, shared);
    p.mark();
    for (HANDLE h : handles)
        dmn_shared_handle_close(h);
    return true;
}

bool c11_fence(Probe& p, bool shared) {
    if (!g.d11_5) {
        p.say("ID3D11Device5 unavailable");
        return false;
    }
    std::vector<Com<ID3D11Fence>> v(g_n);
    for (int i = 0; i < g_n; i++) {
        HRESULT hr = g.d11_5->CreateFence(0, shared ? D3D11_FENCE_FLAG_SHARED
                                                    : D3D11_FENCE_FLAG_NONE,
                                          __uuidof(ID3D11Fence), (void**)&v[i]);
        if (unimpl(p, hr))
            return false;
        if (FAILED(hr) || !v[i])
            return false;
    }
    p.expect((uint64_t)g_n * 16384, 0, shared);
    p.mark();
    return true;
}

/* == Reporting ============================================================ */

/* Fence rows carry a page-sized payload; a ratio against that is meaningless,
 * so they are reported in absolute MB only. */
bool is_ratio_case(const Row& r) { return r.expected >= (4ull << 20); }

/* The verdict is on RESIDENT bytes: memory the machine is actually holding is
 * what slows everything else down, and a reservation nothing writes costs none
 * of it. Reserved bytes are still measured and printed — a 2x reservation can
 * surface in a video-memory budget — but do not fail the test on their own.
 * COPIES outranks both: an import that produced more shm objects than there
 * are producers did not alias at all, which would otherwise hide inside a
 * 2.0 ratio.
 *
 * Unshared controls are never judged: they live inside D3DMetal's pool, whose
 * growth is quantized and whose padding for tiled textures is its own
 * business. They are here to show what the same create costs without us. */
const char* verdict_of(const Row& r, double* ratio) {
    if (!r.ran || !r.marked)
        return "skip";
    *ratio = r.rslope;
    if (!is_ratio_case(r))
        return "n/a";
    if (r.wants_shm && r.shm_objs == 0)
        return "UNSHARED";
    if (r.shm_obj > (int64_t)r.expected + (int64_t)(g_payload / 2))
        return "COPIES";
    if (!r.touched)
        return "untouched"; /* nothing wrote it, so resident says nothing */
    if (!r.wants_shm)
        return *ratio >= g_max_resident ? "pool" : "ok";
    /* Framework pools this kind (its control is `pool`): a row that would
     * otherwise be a finding is reported as `pooled`, not gated — see
     * run_case(). A row that passes strictly keeps its `ok`. */
    bool control_pools = false;
    if (!r.control.empty()) {
        for (const Row& c : g_rows) {
            if (c.name != r.control || c.group != r.group)
                continue;
            control_pools = c.ran && c.marked && c.touched && !c.wants_shm &&
                            c.rslope >= g_max_resident;
            break;
        }
    }
    if (control_pools && (*ratio >= g_max_resident || r.slope >= g_max_ratio))
        return "pooled";
    /* A finding needs BOTH numbers. Resident is host-wide and therefore noisy
     * (other processes move it); reserved is exact, and resident can never
     * physically exceed it, so a real doubling must show in both. */
    if (*ratio >= g_max_resident && r.slope >= g_max_resident)
        return "RAM-DOUBLE";
    if (*ratio >= g_max_resident)
        return "noisy"; /* resident says yes, reserved says no: not believable */
    if (r.slope >= g_max_ratio)
        return "reserve2x"; /* virtual only — reported, not fatal */
    return "ok";
}

bool is_finding(const char* v) {
    return !strcmp(v, "RAM-DOUBLE") || !strcmp(v, "COPIES") ||
           !strcmp(v, "UNSHARED");
}

int report() {
    printf("\n" T_TAG ": ==== bytes of backing per path ====\n");
    printf(T_TAG ": device %s, %d x %.0f MiB held per case\n",
           dmn_test_metal_device_name(), g_n, mib((int64_t)g_payload));
    printf(T_TAG ": rsvd/res = bytes RESERVED / bytes the machine actually "
           "HOLDS, each per byte\n");
    printf(T_TAG ": of resource, as a slope over two payloads so fixed "
           "overheads cancel. The\n");
    printf(T_TAG ": verdict is on res: reserved-but-never-written costs address "
           "space, not RAM.\n");
    printf(T_TAG ": %-8s %-34s %8s %8s %5s %5s %4s  %s\n", "group", "case",
           "residMB", "shmobjMB", "rsvd", "res", "#obj", "verdict");

    int findings = 0;
    for (const Row& r : g_rows) {
        double ratio = 0;
        const char* v = verdict_of(r, &ratio);
        char rs[16] = "-", rr[16] = "-";
        if (r.ran && r.marked) {
            snprintf(rs, sizeof rs, "%.2f", r.slope);
            snprintf(rr, sizeof rr, "%.2f", r.rslope);
        }
        printf(T_TAG ": %-8s %-34s %+8.0f %+8.0f %5s %5s %4d  %s%s%s\n",
               r.group.c_str(), r.name.c_str(), mib(r.resident),
               mib(r.shm_obj), rs, rr, r.shm_objs, v,
               r.note.empty() ? "" : " — ", r.note.c_str());
        if (is_finding(v))
            findings++;
    }

    printf("\n" T_TAG ": ---- findings ----\n");
    if (!findings) {
        printf(T_TAG ": every path backs its resources exactly once.\n");
    } else {
        for (const Row& r : g_rows) {
            double ratio = 0;
            const char* v = verdict_of(r, &ratio);
            if (!is_finding(v))
                continue;
            if (!strcmp(v, "UNSHARED"))
                printf(T_TAG ": UNSHARED: %s/%s asked for a shared resource "
                       "but no shared backing was substituted — the create "
                       "silently produced a private resource\n",
                       r.group.c_str(), r.name.c_str());
            else if (!strcmp(v, "COPIES"))
                printf(T_TAG ": COPIES: %s/%s made %d shm objects (%.0f MB) "
                       "for %.0f MB of resources — it is not aliasing, so "
                       "these duplicate backings are RESIDENT memory\n",
                       r.group.c_str(), r.name.c_str(), r.shm_objs,
                       mib(r.shm_obj), mib((int64_t)r.expected));
            else
                printf(T_TAG ": RAM-DOUBLE: %s/%s holds %.2f bytes of RESIDENT "
                       "memory per byte of resource (%.0f MB for %.0f MB of "
                       "resources) — this is memory the machine cannot use for "
                       "anything else\n",
                       r.group.c_str(), r.name.c_str(), ratio, mib(r.resident),
                       mib((int64_t)r.expected));
        }
    }
    /* The gate. Over-allocation (DOUBLE) and a non-aliasing import (COPIES)
     * fail: both mean an app pays real memory twice. UNSHARED is a capability
     * gap rather than a memory bug — the create succeeded, it just was not
     * substituted — so it warns unless DMN_FP_STRICT asks for everything. */
    int hard = 0, warn = 0, virt = 0;
    for (const Row& r : g_rows) {
        double ratio = 0;
        const char* v = verdict_of(r, &ratio);
        if (!strcmp(v, "RAM-DOUBLE") || !strcmp(v, "COPIES"))
            hard++;
        else if (!strcmp(v, "UNSHARED"))
            warn++;
        else if (!strcmp(v, "reserve2x"))
            virt++;
    }
    int pooled = 0;
    for (const Row& r : g_rows) {
        double ratio = 0;
        if (!strcmp(verdict_of(r, &ratio), "pooled"))
            pooled++;
    }
    if (pooled) {
        printf(T_TAG ": %d shared path(s) are `pooled`: this D3DMetal parks "
               "EVERY resource of that kind in a pool slot it makes resident "
               "(the unshared control reads >= %.2fx), and the substitution "
               "cannot give the slot back — so a shared resource costs slot + "
               "shm, and each open/alias another slot. Reported, not gated; "
               "GPTk 4.0 allocates dedicated and reads 1x.\n",
               pooled, g_max_resident);
    }
    if (virt) {
        printf(T_TAG ": %d path(s) reserve >= %.2fx but hold ~1x resident "
               "(`reserve2x`): D3DMetal's heap allocation goes unused once "
               "placements are backed by the heap's shared object, and nothing "
               "ever writes it. Address space and video-memory budget only.\n",
               virt, g_max_ratio);
    }
    if (warn && !g_strict)
        printf(T_TAG ": %d warning(s) above are not gated; DMN_FP_STRICT=1 "
               "fails on them too.\n", warn);
    if (g_report) {
        if (hard || warn)
            printf(T_TAG ": DMN_FP_REPORT set — %d finding(s) not treated as "
                   "failures.\n", hard + warn);
        return 0;
    }
    const int fatal = hard + (g_strict ? warn : 0);
    fflush(stdout); /* stdout is block-buffered to a file, stderr is not: without
                     * this the failure line lands in the middle of the table */
    if (fatal) {
        fprintf(stderr, T_TAG ": FAIL — %d path(s) hold at or above %.2f bytes "
                "of RESIDENT memory per byte of resource, or do not alias.\n",
                fatal, g_max_resident);
        return 1;
    }
    return 0;
}

} /* namespace */

int main() {
    if (const char* e = getenv("DMN_FP_MB"))
        g_payload = (uint64_t)strtoull(e, nullptr, 10) << 20;
    if (const char* e = getenv("DMN_FP_N"))
        g_n = atoi(e);
    if (const char* e = getenv("DMN_FP_ONLY"))
        g_only = e;
    g_strict = getenv("DMN_FP_STRICT") != nullptr;
    g_report = getenv("DMN_FP_REPORT") != nullptr;
    if (const char* e = getenv("DMN_FP_MAX"))
        g_max_ratio = strtod(e, nullptr);
    if (const char* e = getenv("DMN_FP_MAX_RESIDENT"))
        g_max_resident = strtod(e, nullptr);
    if (!(g_max_ratio > 1.0)) {
        fprintf(stderr, T_TAG ": DMN_FP_MAX must be > 1.0\n");
        return 1;
    }

    if (g_n < 2 || g_payload < (4ull << 20) || g_payload % (4ull * kTexW)) {
        fprintf(stderr, T_TAG ": DMN_FP_N >= 2 and DMN_FP_MB >= 4, a multiple "
                "of 16\n");
        return 1;
    }
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }
    if (FAILED(make_d3d11_device(g.d11, g.ctx11,
                                 D3D11_CREATE_DEVICE_BGRA_SUPPORT))) {
        fprintf(stderr, T_TAG ": D3D11CreateDevice FAILED\n");
        return 1;
    }
    g.d11->QueryInterface(__uuidof(ID3D11Device1), (void**)&g.d11_1);
    g.d11->QueryInterface(__uuidof(ID3D11Device3), (void**)&g.d11_3);
    g.d11->QueryInterface(__uuidof(ID3D11Device5), (void**)&g.d11_5);
    if (FAILED(make_d3d12_device(g.d12))) {
        fprintf(stderr, T_TAG ": D3D12CreateDevice FAILED\n");
        return 1;
    }
    /* Copy source + submission plumbing for the residency touches. The source
     * is 2x the payload because the second pass doubles it, and it is written
     * once HERE so its own pages are inside every case's baseline. */
    if (FAILED(make_d3d12_queue(g.d12.ptr(), g.q12)) ||
        FAILED(g.d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             __uuidof(ID3D12CommandAllocator),
                                             (void**)&g.alloc12)) ||
        FAILED(g.d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        g.alloc12.ptr(), nullptr,
                                        __uuidof(ID3D12GraphicsCommandList),
                                        (void**)&g.list12)) ||
        FAILED(g.d12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  __uuidof(ID3D12Fence), (void**)&g.fence12))) {
        fprintf(stderr, T_TAG ": D3D12 submission setup FAILED\n");
        return 1;
    }
    g.list12->Close();
    {
        D3D12_RESOURCE_DESC sd = buf_desc(g_payload * 2);
        if (FAILED(committed(D3D12_HEAP_TYPE_DEFAULT, sd, D3D12_HEAP_FLAG_NONE,
                             &g.src12)) || !g.src12) {
            fprintf(stderr, T_TAG ": D3D12 copy source FAILED\n");
            return 1;
        }
        std::vector<Com<ID3D12Resource>> warm{g.src12};
        touch12_bufs(warm, g_payload * 2);
    }

    using D = D3D12_HEAP_FLAGS;
    const D kNone = D3D12_HEAP_FLAG_NONE, kShared = D3D12_HEAP_FLAG_SHARED;

    run_case("d3d11", "CreateTexture2D (control)",
             [](Probe& p) { return c11_tex(p, 0); });
    run_case("d3d11", "CreateTexture2D SHARED",
             [](Probe& p) { return c11_tex(p, D3D11_RESOURCE_MISC_SHARED); },
             "CreateTexture2D (control)");
    run_case("d3d11", "CreateTexture2D KEYEDMUTEX", [](Probe& p) {
        return c11_tex(p, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX);
    }, "CreateTexture2D (control)");
    run_case("d3d11", "CreateTexture2D1 SHARED", c11_tex1,
             "CreateTexture2D (control)");
    run_case("d3d11", "CreateBuffer (control)",
             [](Probe& p) { return c11_buf(p, 0); });
    run_case("d3d11", "CreateBuffer SHARED",
             [](Probe& p) { return c11_buf(p, D3D11_RESOURCE_MISC_SHARED); },
             "CreateBuffer (control)");
    run_case("d3d11", "CreateTexture3D (control)",
             [](Probe& p) { return c11_tex3d(p, 0); });
    run_case("d3d11", "CreateTexture3D SHARED", [](Probe& p) {
        return c11_tex3d(p, D3D11_RESOURCE_MISC_SHARED);
    });
    run_case("d3d11", "GetSharedHandle x1",
             [](Probe& p) { return c11_export(p, false, 1); });
    run_case("d3d11", "CreateSharedHandle x1",
             [](Probe& p) { return c11_export(p, true, 1); });
    run_case("d3d11", "CreateSharedHandle x4",
             [](Probe& p) { return c11_export(p, true, 4); });
    run_case("d3d11", "OpenSharedResource x1",
             [](Probe& p) { return c11_open(p, 1, false); });
    run_case("d3d11", "OpenSharedResource x4",
             [](Probe& p) { return c11_open(p, 4, false); });
    run_case("d3d11", "OpenSharedResource1 x1",
             [](Probe& p) { return c11_open(p, 1, true); });

    run_case("d3d12", "CreateCommittedResource buf (ctl)",
             [&](Probe& p) { return c12_committed(p, kNone, false); });
    run_case("d3d12", "CreateCommittedResource buf SHARED",
             [&](Probe& p) { return c12_committed(p, kShared, false); },
             "CreateCommittedResource buf (ctl)");
    run_case("d3d12", "CreateCommittedResource tex (ctl)",
             [&](Probe& p) { return c12_committed(p, kNone, true); });
    run_case("d3d12", "CreateCommittedResource tex SHARED",
             [&](Probe& p) { return c12_committed(p, kShared, true); },
             "CreateCommittedResource tex (ctl)");
    run_case("d3d12", "CreateCommittedResource1 SHARED",
             [&](Probe& p) { return c12_committed_v(p, 1, kShared); });
    run_case("d3d12", "CreateCommittedResource2 SHARED",
             [&](Probe& p) { return c12_committed_v(p, 2, kShared); });
    run_case("d3d12", "CreateCommittedResource3 SHARED",
             [&](Probe& p) { return c12_committed_v(p, 3, kShared); });
    run_case("d3d12", "CreateHeap only (control)",
             [&](Probe& p) { return c12_heap_only(p, kNone); });
    run_case("d3d12", "CreateHeap only SHARED",
             [&](Probe& p) { return c12_heap_only(p, kShared); });
    run_case("d3d12", "CreateHeap +placed buf (control)",
             [&](Probe& p) { return c12_heap_placed(p, kNone, false, 0, 0); });
    run_case("d3d12", "CreateHeap +placed buf SHARED",
             [&](Probe& p) { return c12_heap_placed(p, kShared, false, 0, 0); },
             "CreateHeap +placed buf (control)");
    run_case("d3d12", "CreateHeap +placed tex (control)",
             [&](Probe& p) { return c12_heap_placed(p, kNone, true, 0, 0); });
    run_case("d3d12", "CreateHeap +placed tex SHARED",
             [&](Probe& p) { return c12_heap_placed(p, kShared, true, 0, 0); },
             "CreateHeap +placed tex (control)");
    run_case("d3d12", "CreateHeap1 +placed buf SHARED",
             [&](Probe& p) { return c12_heap_placed(p, kShared, false, 1, 0); },
             "CreateHeap +placed buf (control)");
    run_case("d3d12", "CreatePlacedResource1 SHARED",
             [&](Probe& p) { return c12_heap_placed(p, kShared, false, 0, 1); },
             "CreateHeap +placed buf (control)");
    run_case("d3d12", "CreatePlacedResource2 SHARED",
             [&](Probe& p) { return c12_heap_placed(p, kShared, false, 0, 2); },
             "CreateHeap +placed buf (control)");
    run_case("d3d12", "placed x4 same offset (alias)",
             [](Probe& p) { return c12_heap_alias(p, 4); },
             "CreateHeap +placed buf (control)");
    run_case("d3d12", "CreateSharedHandle x1",
             [](Probe& p) { return c12_export(p, 1); },
             "CreateCommittedResource buf (ctl)");
    run_case("d3d12", "CreateSharedHandle x4",
             [](Probe& p) { return c12_export(p, 4); },
             "CreateCommittedResource buf (ctl)");
    run_case("d3d12", "OpenSharedHandle x1",
             [](Probe& p) { return c12_open(p, 1); },
             "CreateCommittedResource buf (ctl)");
    run_case("d3d12", "OpenSharedHandle x4",
             [](Probe& p) { return c12_open(p, 4); },
             "CreateCommittedResource buf (ctl)");

    run_case("xapi", "d3d11 tex -> d3d12 open", c_xapi_11_to_12);
    run_case("xapi", "d3d12 tex -> d3d11 open", c_xapi_12_to_11);

    run_case("import", "heap_from_fd only",
             [](Probe& p) { return c_heap_import(p, false); });
    /* No control in its group, so this row is judged strictly. */
    run_case("import", "heap_from_fd +placed buf",
             [](Probe& p) { return c_heap_import(p, true); });

    run_case("fence", "d3d12 CreateFence (control)",
             [](Probe& p) { return c12_fence(p, false, false); });
    run_case("fence", "d3d12 CreateFence SHARED",
             [](Probe& p) { return c12_fence(p, true, false); });
    run_case("fence", "d3d12 CreateFence SHARED +export",
             [](Probe& p) { return c12_fence(p, true, true); });
    run_case("fence", "d3d11 CreateFence (control)",
             [](Probe& p) { return c11_fence(p, false); });
    run_case("fence", "d3d11 CreateFence SHARED",
             [](Probe& p) { return c11_fence(p, true); });

    /* Child mode: run the one named case in a fresh process, GPU-write what it
     * made, and report what the machine's resident page count did. */
    if (const char* want = getenv("DMN_FP_CASE")) {
        for (const CaseDef& c : g_cases) {
            if (c.name != want)
                continue;
            g_do_touch = true;
            settle_host();
            if (memory_pressure())
                return 0; /* no RESIDENT_BYTES line: parent records untouched */
            Probe p;
            p.base_shm = shm_census();
            p.base_host = host_used();
            p.base = metal_alloc();
            const bool ran = c.fn(p);
            if (ran && p.touched)
                printf("%s%lld\n", kResidentPrefix, (long long)p.resident);
            fflush(stdout);
            return 0;
        }
        return 0;
    }

    for (const CaseDef& c : g_cases)
        measure_case(c);

    const int rc = report();
    if (rc == 0)
        T_PASS();
    return rc;
}
