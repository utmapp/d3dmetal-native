/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTPath — Windows<->Unix path mapping and executable/module path
 * queries. All wide strings are char16_t (16-bit, Windows ABI); never
 * wchar_t, which is 32-bit on macOS.
 */

#include <dlfcn.h>
#include <mach-o/dyld.h>

#include <cstring>
#include <mutex>
#include <string>

#include "dmn_gfxt.h"
#include "dmn_private.h"

namespace {

/* dmn_set_executable_path override; D3DMetal's per-app profile matcher keys
 * on what getExecutablePath reports, and embedders acting on behalf of
 * another program (VM display servers, remoting hosts) want that program's
 * path, not their own. */
std::mutex g_exe_path_mutex;
std::string g_exe_path_override;

bool exe_path_override(char* out, uint32_t bytes) {
    std::lock_guard<std::mutex> lock(g_exe_path_mutex);
    if (g_exe_path_override.empty())
        return false;
    size_t n = g_exe_path_override.size();
    if (n >= bytes)
        n = bytes - 1;
    memcpy(out, g_exe_path_override.c_str(), n);
    out[n] = '\0';
    return true;
}

/* Minimal UTF-16 <-> UTF-8 conversion (with surrogate pairs). */

std::string utf16_to_utf8(const char16_t* in) {
    std::string out;
    if (!in)
        return out;
    for (size_t i = 0; in[i]; i++) {
        uint32_t cp = in[i];
        if (cp >= 0xd800 && cp <= 0xdbff && in[i + 1] >= 0xdc00 &&
            in[i + 1] <= 0xdfff) {
            cp = 0x10000 + ((cp - 0xd800) << 10) + (in[i + 1] - 0xdc00);
            i++;
        }
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xc0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3f));
        } else if (cp < 0x10000) {
            out += (char)(0xe0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3f));
            out += (char)(0x80 | (cp & 0x3f));
        } else {
            out += (char)(0xf0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3f));
            out += (char)(0x80 | ((cp >> 6) & 0x3f));
            out += (char)(0x80 | (cp & 0x3f));
        }
    }
    return out;
}

std::u16string utf8_to_utf16(const char* in) {
    std::u16string out;
    if (!in)
        return out;
    const auto* p = (const unsigned char*)in;
    while (*p) {
        uint32_t cp = 0;
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p >> 5) == 0x6 && p[1]) {
            cp = ((p[0] & 0x1f) << 6) | (p[1] & 0x3f);
            p += 2;
        } else if ((*p >> 4) == 0xe && p[1] && p[2]) {
            cp = ((p[0] & 0x0f) << 12) | ((p[1] & 0x3f) << 6) | (p[2] & 0x3f);
            p += 3;
        } else if ((*p >> 3) == 0x1e && p[1] && p[2] && p[3]) {
            cp = ((p[0] & 0x07) << 18) | ((p[1] & 0x3f) << 12) |
                 ((p[2] & 0x3f) << 6) | (p[3] & 0x3f);
            p += 4;
        } else {
            p++; /* invalid byte; skip */
            continue;
        }
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out += (char16_t)(0xd800 + (cp >> 10));
            out += (char16_t)(0xdc00 + (cp & 0x3ff));
        } else {
            out += (char16_t)cp;
        }
    }
    return out;
}

} // namespace

DmnGFXTPath::~DmnGFXTPath() = default;

bool DmnGFXTPath::windowsToUnixPath(const char16_t* in, char* out,
                                    size_t* inOutBytes) {
    if (!in || !out || !inOutBytes)
        return false;

    std::string s = utf16_to_utf8(in);
    for (char& c : s) {
        if (c == '\\')
            c = '/';
    }

    /* The Windows system directory maps onto the framework's bundled
     * Resources (system32 is where D3DMetal would look for its DLLs). */
    static const char* kSys32 = "c:/windows/system32";
    std::string lower = s;
    for (char& c : lower)
        c = (char)tolower((unsigned char)c);
    if (lower.compare(0, strlen(kSys32), kSys32) == 0) {
        s = dmn_framework_resources_dir() + s.substr(strlen(kSys32));
    } else if (s.size() >= 2 && s[1] == ':') {
        s = s.substr(2); /* strip drive letter ("Z:/foo" -> "/foo") */
    }

    DMN_TRACE("path: windowsToUnixPath -> %s", s.c_str());

    size_t needed = s.size() + 1;
    if (*inOutBytes < needed) {
        *inOutBytes = needed;
        return false;
    }
    memcpy(out, s.c_str(), needed);
    *inOutBytes = needed;
    return true;
}

bool DmnGFXTPath::unixToWindowsPath(const char* in, char16_t* out,
                                    size_t* inOutChars) {
    if (!in || !out || !inOutChars)
        return false;

    /* Wine convention: the root filesystem appears as drive Z:. */
    std::string s = std::string("Z:") + in;
    for (char& c : s) {
        if (c == '/')
            c = '\\';
    }
    std::u16string w = utf8_to_utf16(s.c_str());

    DMN_TRACE("path: unixToWindowsPath(%s)", in);

    size_t needed = w.size() + 1;
    if (*inOutChars < needed) {
        *inOutChars = needed;
        return false;
    }
    memcpy(out, w.c_str(), needed * sizeof(char16_t));
    *inOutChars = needed;
    return true;
}

bool DmnGFXTPath::windowsSystemDirectoryPath(char16_t* out,
                                             size_t* inOutChars) {
    static const char16_t kPath[] = u"C:\\windows\\system32";
    const size_t needed = sizeof(kPath) / sizeof(kPath[0]); /* incl. NUL */
    if (!out || !inOutChars)
        return false;
    if (*inOutChars < needed) {
        *inOutChars = needed;
        return false;
    }
    memcpy(out, kPath, sizeof(kPath));
    *inOutChars = needed;
    return true;
}

void DmnGFXTPath::getExecutablePath(char* out, uint32_t bytes) {
    if (!out || bytes == 0)
        return;
    if (exe_path_override(out, bytes)) {
        DMN_TRACE("path: getExecutablePath (override) -> %s", out);
        return;
    }
    uint32_t want = bytes;
    if (_NSGetExecutablePath(out, &want) != 0)
        out[0] = '\0';
    DMN_TRACE("path: getExecutablePath -> %s", out);
}

extern "C" dmn_result dmn_set_executable_path(const char* path) {
    std::lock_guard<std::mutex> lock(g_exe_path_mutex);
    g_exe_path_override = path ? path : "";
    DMN_INFO("path: executable path override %s%s",
             path ? "= " : "cleared", path ? path : "");
    return DMN_SUCCESS;
}

void DmnGFXTPath::getModulePath(void* module, char* out, uint32_t bytes) {
    if (!out || bytes == 0)
        return;
    if (!module) {
        getExecutablePath(out, bytes);
        return;
    }
    Dl_info info{};
    if (dladdr(module, &info) && info.dli_fname) {
        size_t n = strlen(info.dli_fname);
        if (n >= bytes)
            n = bytes - 1;
        memcpy(out, info.dli_fname, n);
        out[n] = '\0';
    } else {
        out[0] = '\0';
    }
    DMN_TRACE("path: getModulePath(%p) -> %s", module, out);
}
