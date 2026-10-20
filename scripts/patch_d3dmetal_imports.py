#!/usr/bin/env python3
"""Rebind D3DMetal's flagged private-API imports to d3dmetal-native's public-API shims.

The imports stay ordinary, visibly named undefined symbols; they just resolve to
src/dmn_import_shims.mm instead of the private Apple symbols.  Nothing is hidden:
`nm -u`, `otool -L` and `dyld_info -fixups` show exactly what the binary imports.

  patch_d3dmetal_imports.py patch <D3DMetal> --provider <install-name>
                                  [--check-provider <d3dmetal-native>] [--sign ID|none]
  patch_d3dmetal_imports.py check <D3DMetal> [--check-provider <d3dmetal-native>]

`check` exits non-zero while any flagged import or private load command remains,
or when a replacement is bound anywhere but d3dmetal-native; CI gates on it.
--check-provider also requires that library to export every bound replacement.
The patch re-signs the binary alone (ad-hoc by default); a caller that ships the
framework re-signs the bundle afterwards.

Only thin x86_64/arm64 Mach-O with LC_DYLD_CHAINED_FIXUPS, imports_format=1 and
uncompressed symbols (symbols_format=0) is supported; anything else is refused.
Each rename is done in place, so a replacement must be no longer than the original.
"""
import argparse
import os
import struct
import subprocess
import sys

RENAMES = {
    "_AnalyticsSendEventLazy": "_dmn_analytics_send",
    "_MTLSetShaderCachePath": "_dmn_metal_cache_set",
    "_MTLMakeShaderCacheWritableByAllUsers": "_dmn_metal_cache_make_shared",
    "_MTLTextureSwizzleChannelsToKey": "_dmn_swizzle_channels_to_key",
    "_MTLTextureSwizzleKeyToChannels": "_dmn_swizzle_key_to_channels",
    "__dyld_get_image_uuid": "_dmn_image_uuid",
    "_dyld_image_header_containing_address": "_dmn_image_header_containing",
    "_OBJC_CLASS_$_MTLAccelerationStructureAllocationDescriptor":
        "_OBJC_CLASS_$_DMNAccelerationStructureAllocationDescriptor",
}
PRIVATE_LIB = "/System/Library/PrivateFrameworks/CoreAnalytics.framework/Versions/A/CoreAnalytics"

MH_MAGIC_64 = 0xFEEDFACF
LC_SYMTAB = 0x2
LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB = 0xC, 0x80000018, 0x8000001F
LC_LOAD_UPWARD_DYLIB, LC_LAZY_LOAD_DYLIB = 0x80000023, 0x20
DYLIB_CMDS = {LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_LOAD_UPWARD_DYLIB, LC_LAZY_LOAD_DYLIB}
LC_DYLD_CHAINED_FIXUPS = 0x80000034


class Refuse(Exception):
    pass


def cstr(buf, off):
    return buf[off:buf.index(b"\0", off)].decode()


class Image:
    def __init__(self, data):
        self.d = bytearray(data)
        magic, _, _, _, ncmds, _, _, _ = struct.unpack_from("<8I", self.d, 0)
        if magic != MH_MAGIC_64:
            raise Refuse("not a thin 64-bit little-endian Mach-O")
        self.dylibs, self.symtab, self.fixups = [], None, None
        off = 32
        for _ in range(ncmds):
            cmd, size = struct.unpack_from("<2I", self.d, off)
            if cmd in DYLIB_CMDS:
                name_off = struct.unpack_from("<I", self.d, off + 8)[0]
                self.dylibs.append((off, size, name_off, cstr(self.d, off + name_off)))
            elif cmd == LC_SYMTAB:
                self.symtab = struct.unpack_from("<4I", self.d, off + 8)  # symoff nsyms stroff strsize
            elif cmd == LC_DYLD_CHAINED_FIXUPS:
                self.fixups = struct.unpack_from("<2I", self.d, off + 8)  # dataoff datasize
            off += size
        if not self.symtab or not self.fixups:
            raise Refuse("missing LC_SYMTAB or LC_DYLD_CHAINED_FIXUPS")
        base = self.fixups[0]
        (_, _, self.imp_off, self.sym_off, self.imp_cnt, fmt, symfmt) = struct.unpack_from("<7I", self.d, base)
        if fmt != 1 or symfmt != 0:
            raise Refuse(f"unsupported imports_format={fmt} symbols_format={symfmt}")
        self.imp_base, self.pool = base + self.imp_off, base + self.sym_off

    def imports(self):
        """[(index, lib_ordinal, name_offset, name)]"""
        out = []
        for i in range(self.imp_cnt):
            v = struct.unpack_from("<I", self.d, self.imp_base + 4 * i)[0]
            out.append((i, v & 0xFF, v >> 9, cstr(self.d, self.pool + (v >> 9))))
        return out

    def nlists(self):
        symoff, nsyms, stroff, _ = self.symtab
        for i in range(nsyms):
            n_strx, n_type, _, n_desc, _ = struct.unpack_from("<IBBHQ", self.d, symoff + 16 * i)
            yield symoff + 16 * i, n_strx, n_type, n_desc

    def provider_ordinal(self):
        for i, (_, _, _, name) in enumerate(self.dylibs, 1):
            if name == PRIVATE_LIB:
                return i
        return None


def rename_in_place(d, base, start, old, new, others):
    """Overwrite the NUL-terminated string at `start` in the table at `base`; refuse if
    another string's characters overlap it (linker tail merging, either direction)."""
    if len(new) > len(old):
        raise Refuse(f"replacement {new} is longer than {old}")
    if any(start < o <= start + len(old) for o in others if o != start):
        raise Refuse(f"{old} shares a linker-merged suffix with a neighbouring symbol")
    if start > base and d[start - 1] != 0:
        raise Refuse(f"{old} is the merged tail of a longer string")
    d[start:start + len(old) + 1] = new.encode() + b"\0" * (len(old) - len(new) + 1)


def provider_exports(lib):
    out = subprocess.run(["nm", "-gU", "-arch", "x86_64", lib], check=True,
                         capture_output=True, text=True).stdout
    return {line.split()[-1] for line in out.splitlines() if line.strip()}


def patch(path, provider, sign):
    img = Image(open(path, "rb").read())
    ordinal = img.provider_ordinal()
    if ordinal is None:
        raise Refuse("CoreAnalytics load command not found (already patched, or an unknown D3DMetal)")
    imps = img.imports()
    found = {n for _, _, _, n in imps} & set(RENAMES)
    if not found:
        raise Refuse("none of the flagged imports are present; unknown D3DMetal")
    for n in sorted(set(RENAMES) - found):
        print(f"note: {n} not imported by this version, skipping")
    todo = {n: RENAMES[n] for n in found}
    pool_offs = [o for _, _, o, _ in imps]
    for i, _, off, name in imps:
        if name in todo:
            rename_in_place(img.d, img.pool, img.pool + off, name, todo[name], [img.pool + o for o in pool_offs])
            v = struct.unpack_from("<I", img.d, img.imp_base + 4 * i)[0]
            struct.pack_into("<I", img.d, img.imp_base + 4 * i, (v & ~0xFF) | ordinal)

    stroff = img.symtab[2]
    nl = list(img.nlists())
    str_starts = [stroff + s for _, s, _, _ in nl]
    done = set()
    for entry, strx, ntype, desc in nl:
        name = cstr(img.d, stroff + strx)
        if name in todo and ntype & 0x0E == 0 and entry not in done:  # N_UNDF | N_EXT
            rename_in_place(img.d, stroff, stroff + strx, name, todo[name], str_starts)
            struct.pack_into("<H", img.d, entry + 6, (desc & 0x00FF) | (ordinal << 8))
            done.add(entry)
    if len(done) != len(todo):
        raise Refuse(f"symtab renamed {len(done)} of {len(todo)} symbols")
    tmp = path + ".patching"
    with open(tmp, "wb") as f:
        f.write(img.d)
    os.chmod(tmp, os.stat(path).st_mode)
    subprocess.check_call(["install_name_tool", "-change", PRIVATE_LIB, provider, tmp],
                          stderr=subprocess.DEVNULL)  # "will invalidate the code signature"
    if sign != "none":
        subprocess.check_call(["codesign", "--force", "--sign", sign, tmp])
    os.replace(tmp, path)
    print(f"patched {path}: {len(todo)} imports -> {provider} (ordinal {ordinal})")


def check(path, provider_lib=None):
    img = Image(open(path, "rb").read())
    bad = []
    for d in img.dylibs:
        if "/PrivateFrameworks/" in d[3]:
            bad.append("load command " + d[3])
    imps = img.imports()
    names = {n for _, _, _, n in imps}
    stroff = img.symtab[2]
    names |= {cstr(img.d, stroff + s) for _, s, _, _ in img.nlists()}
    bad += [f"import {n}" for n in sorted(names & set(RENAMES))]

    bound = {}
    for _, ordinal, _, name in imps:
        if name in RENAMES.values():
            lib = img.dylibs[ordinal - 1][3] if 0 < ordinal <= len(img.dylibs) else f"ordinal {ordinal}"
            bound[name] = lib
            if lib.startswith(("/System/", "/usr/")):
                bad.append(f"{name} bound to {lib}")
    if not bound:
        bad.append("no replacement is bound (unpatched?)")
    if provider_lib and bound:
        missing = sorted(set(bound) - provider_exports(provider_lib))
        bad += [f"{provider_lib} does not export {n}" for n in missing]
    if bad:
        print(f"FAIL {path}:\n  " + "\n  ".join(bad))
        return 1
    print(f"OK {path}: {len(bound)} replacements bound to {sorted(set(bound.values()))}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["patch", "check"])
    ap.add_argument("binary")
    ap.add_argument("--provider", help="install name of d3dmetal-native as D3DMetal should reference it")
    ap.add_argument("--check-provider", metavar="DYLIB",
                    help="d3dmetal-native build that must export every bound replacement")
    ap.add_argument("--sign", default="-", help="codesign identity, or 'none' (default: ad-hoc)")
    a = ap.parse_args()
    try:
        if a.mode == "check":
            return check(a.binary, a.check_provider)
        if not a.provider:
            ap.error("patch needs --provider")
        if a.check_provider:
            img = Image(open(a.binary, "rb").read())
            wanted = {RENAMES[n] for _, _, _, n in img.imports() if n in RENAMES}
            missing = sorted(wanted - provider_exports(a.check_provider))
            if missing:
                raise Refuse(f"{a.check_provider} does not export {missing}")
        patch(a.binary, a.provider, a.sign)
        return check(a.binary, a.check_provider)
    except Refuse as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
