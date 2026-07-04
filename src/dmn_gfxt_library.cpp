/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTLibrary — dlopen/dlsym-backed library loading. Windows DLL
 * names map onto the framework's bundled Resources dylibs
 * (dxcompiler.dll -> libdxcompiler.dylib etc.). Every request is logged
 * at INFO so unmapped DLL names surface in the logs.
 */

#include <dlfcn.h>

#include <cctype>
#include <cstring>
#include <string>

#include "dmn_gfxt.h"
#include "dmn_private.h"

namespace {

/* "C:\foo\DXCompiler.DLL" -> stem "dxcompiler", had_dll = true */
std::string dll_stem(const char* name, bool* had_dll) {
    std::string s = name ? name : "";
    size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos)
        s = s.substr(slash + 1);
    for (char& c : s)
        c = (char)tolower((unsigned char)c);
    *had_dll = false;
    if (s.size() > 4 && s.compare(s.size() - 4, 4, ".dll") == 0) {
        s = s.substr(0, s.size() - 4);
        *had_dll = true;
    }
    return s;
}

void* open_with_mapping(const char* name, int mode) {
    if (!name || !*name)
        return nullptr;

    bool had_dll = false;
    std::string stem = dll_stem(name, &had_dll);
    const std::string& resources = dmn_framework_resources_dir();

    if (!resources.empty() && !stem.empty()) {
        for (const std::string& candidate :
             {resources + "/lib" + stem + ".dylib",
              resources + "/" + stem + ".dylib"}) {
            if (void* h = dlopen(candidate.c_str(), mode)) {
                DMN_INFO("library: %s -> %s", name, candidate.c_str());
                return h;
            }
        }
    }

    if (void* h = dlopen(name, mode)) {
        DMN_INFO("library: %s -> dlopen as-is", name);
        return h;
    }

    DMN_INFO("library: %s -> not found (%s)", name,
             dlerror() ?: "no dlerror");
    return nullptr;
}

} // namespace

DmnGFXTLibrary::~DmnGFXTLibrary() = default;

void* DmnGFXTLibrary::loadLibrary(const char* name) {
    DMN_INFO("library: loadLibrary(%s)", name ? name : "(null)");
    return open_with_mapping(name, RTLD_NOW | RTLD_LOCAL);
}

void* DmnGFXTLibrary::getModuleHandle(const char* name) {
    DMN_INFO("library: getModuleHandle(%s)", name ? name : "(null)");
    return open_with_mapping(name, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
}

void* DmnGFXTLibrary::getProcAddress(void* module, const char* name) {
    if (!name)
        return nullptr;
    void* sym = module ? dlsym(module, name)
                       : dlsym(RTLD_DEFAULT, name);
    DMN_TRACE("library: getProcAddress(%p, %s) = %p", module, name, sym);
    return sym;
}

void DmnGFXTLibrary::freeLibrary(void* module) {
    DMN_TRACE("library: freeLibrary(%p)", module);
    if (module)
        dlclose(module);
}

void* DmnGFXTLibrary::loadLibraryFromSystemDirectory(const char* name) {
    DMN_INFO("library: loadLibraryFromSystemDirectory(%s)",
             name ? name : "(null)");
    /* system32 is, for our purposes, the framework Resources directory. */
    return open_with_mapping(name, RTLD_NOW | RTLD_LOCAL);
}
