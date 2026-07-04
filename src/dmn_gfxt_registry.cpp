/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTRegistry — in-memory registry store. OpenKey/CreateKey always
 * succeed (D3DMetal expects to open keys it just created; a missing key
 * simply yields GetValue misses). Every access logs at DEBUG so
 * DMN_LOG=debug reveals which settings D3DMetal consults.
 *
 * Value reads honor DMN_REG_<NAME> environment overrides (name uppercased,
 * non-alphanumerics mapped to '_').
 */

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <variant>

#include "dmn_gfxt.h"
#include "dmn_private.h"

namespace {

using DmnRegValue =
    std::variant<uint32_t, uint64_t, std::string, std::vector<uint8_t>>;

std::mutex g_reg_mutex;
std::map<std::string, DmnRegValue> g_reg_values; /* "<root>\subkey\\name" */

struct DmnRegKey {
    std::string path; /* "<root>\subkey", lowercased */
};

std::string lc(const char* s) {
    std::string out = s ? s : "";
    for (char& c : out)
        c = (char)tolower((unsigned char)c);
    return out;
}

std::string key_path(uint32_t root, const char* subkey) {
    return std::to_string(root) + "\\" + lc(subkey);
}

std::string value_path(const DmnRegKey* key, const char* name) {
    return key->path + "\\\\" + lc(name);
}

const char* env_override(const char* name) {
    if (!name)
        return nullptr;
    std::string var = "DMN_REG_";
    for (const char* p = name; *p; ++p)
        var += isalnum((unsigned char)*p) ? (char)toupper((unsigned char)*p)
                                          : '_';
    return getenv(var.c_str());
}

template <typename T>
bool get_typed(void* key, const char* name, T& out, const char* type_tag) {
    auto* k = static_cast<DmnRegKey*>(key);
    if (!k || !name)
        return false;
    std::lock_guard<std::mutex> lock(g_reg_mutex);
    auto it = g_reg_values.find(value_path(k, name));
    if (it == g_reg_values.end()) {
        DMN_DEBUG("registry: GetValue(%s, %s) [%s] -> miss",
                  k->path.c_str(), name, type_tag);
        return false;
    }
    if (auto* v = std::get_if<T>(&it->second)) {
        out = *v;
        DMN_DEBUG("registry: GetValue(%s, %s) [%s] -> hit",
                  k->path.c_str(), name, type_tag);
        return true;
    }
    DMN_WARN("registry: GetValue(%s, %s) type mismatch (wanted %s)",
             k->path.c_str(), name, type_tag);
    return false;
}

template <typename T>
void set_typed(void* key, const char* name, T value, const char* type_tag) {
    auto* k = static_cast<DmnRegKey*>(key);
    if (!k || !name)
        return;
    DMN_DEBUG("registry: SetValue(%s, %s) [%s]", k->path.c_str(), name,
              type_tag);
    std::lock_guard<std::mutex> lock(g_reg_mutex);
    g_reg_values[value_path(k, name)] = std::move(value);
}

} // namespace

DmnGFXTRegistry::~DmnGFXTRegistry() = default;

void* DmnGFXTRegistry::OpenKey(RegistryMainKey root, const char* subkey) {
    DMN_DEBUG("registry: OpenKey(%u, %s)", (unsigned)root,
              subkey ? subkey : "(null)");
    return new DmnRegKey{key_path(root, subkey)};
}

void* DmnGFXTRegistry::CreateKey(RegistryMainKey root, const char* subkey,
                                 bool unused, bool* createdNew) {
    (void)unused;
    std::string path = key_path(root, subkey);
    DMN_DEBUG("registry: CreateKey(%u, %s)", (unsigned)root,
              subkey ? subkey : "(null)");
    if (createdNew) {
        std::lock_guard<std::mutex> lock(g_reg_mutex);
        auto it = g_reg_values.lower_bound(path);
        *createdNew = !(it != g_reg_values.end() &&
                        it->first.compare(0, path.size(), path) == 0);
    }
    return new DmnRegKey{std::move(path)};
}

void DmnGFXTRegistry::CloseKey(void* key) {
    delete static_cast<DmnRegKey*>(key);
}

void DmnGFXTRegistry::SetValue(void* key, const char* name, uint32_t value) {
    set_typed(key, name, value, "u32");
}

void DmnGFXTRegistry::SetValue(void* key, const char* name, uint64_t value) {
    set_typed(key, name, value, "u64");
}

void DmnGFXTRegistry::SetValue(void* key, const char* name,
                               const std::string& value) {
    set_typed(key, name, value, "str");
}

void DmnGFXTRegistry::SetValue(void* key, const char* name,
                               const std::vector<uint8_t>& value) {
    set_typed(key, name, value, "bin");
}

bool DmnGFXTRegistry::GetValue(void* key, const char* name, uint32_t& out) {
    if (const char* env = env_override(name)) {
        out = (uint32_t)strtoul(env, nullptr, 0);
        DMN_DEBUG("registry: GetValue(%s) [u32] -> env override %s",
                  name, env);
        return true;
    }
    return get_typed(key, name, out, "u32");
}

bool DmnGFXTRegistry::GetValue(void* key, const char* name, uint64_t& out) {
    if (const char* env = env_override(name)) {
        out = strtoull(env, nullptr, 0);
        DMN_DEBUG("registry: GetValue(%s) [u64] -> env override %s",
                  name, env);
        return true;
    }
    return get_typed(key, name, out, "u64");
}

bool DmnGFXTRegistry::GetValue(void* key, const char* name, std::string& out) {
    if (const char* env = env_override(name)) {
        out = env;
        DMN_DEBUG("registry: GetValue(%s) [str] -> env override %s",
                  name, env);
        return true;
    }
    return get_typed(key, name, out, "str");
}

bool DmnGFXTRegistry::GetValue(void* key, const char* name,
                               std::vector<uint8_t>& out) {
    return get_typed(key, name, out, "bin");
}

bool DmnGFXTRegistry::DeleteValue(void* key, const char* name,
                                  const char* unused) {
    (void)unused;
    auto* k = static_cast<DmnRegKey*>(key);
    if (!k || !name)
        return false;
    DMN_DEBUG("registry: DeleteValue(%s, %s)", k->path.c_str(), name);
    std::lock_guard<std::mutex> lock(g_reg_mutex);
    return g_reg_values.erase(value_path(k, name)) > 0;
}

/* == Public pre-seeding API =============================================== */

namespace {

template <typename T>
bool store_value(uint32_t root, const char* subkey, const char* name,
                 T value) {
    if (!subkey || !name)
        return false;
    std::lock_guard<std::mutex> lock(g_reg_mutex);
    g_reg_values[key_path(root, subkey) + "\\\\" + lc(name)] =
        std::move(value);
    return true;
}

} // namespace

bool dmn_registry_store_u32(uint32_t root, const char* subkey,
                            const char* name, uint32_t value) {
    return store_value(root, subkey, name, value);
}

bool dmn_registry_store_u64(uint32_t root, const char* subkey,
                            const char* name, uint64_t value) {
    return store_value(root, subkey, name, value);
}

bool dmn_registry_store_string(uint32_t root, const char* subkey,
                               const char* name, const char* value) {
    return value && store_value(root, subkey, name, std::string(value));
}

extern "C" dmn_result dmn_registry_set_u32(dmn_registry_root root,
                                           const char* subkey,
                                           const char* name, uint32_t value) {
    return dmn_registry_store_u32(root, subkey, name, value)
        ? DMN_SUCCESS : DMN_ERROR_INVALID_ARGUMENT;
}

extern "C" dmn_result dmn_registry_set_u64(dmn_registry_root root,
                                           const char* subkey,
                                           const char* name, uint64_t value) {
    return dmn_registry_store_u64(root, subkey, name, value)
        ? DMN_SUCCESS : DMN_ERROR_INVALID_ARGUMENT;
}

extern "C" dmn_result dmn_registry_set_string(dmn_registry_root root,
                                              const char* subkey,
                                              const char* name,
                                              const char* value) {
    return dmn_registry_store_string(root, subkey, name, value)
        ? DMN_SUCCESS : DMN_ERROR_INVALID_ARGUMENT;
}
