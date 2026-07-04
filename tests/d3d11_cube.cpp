/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Classic spinning 3D cube on D3D11, running natively on macOS via
 * libd3dmetal-native (Cocoa window + D3DMetal.framework). Exercises what
 * the triangle test does not: depth buffering, perspective projection,
 * per-frame constant-buffer updates with a full MVP matrix, and 3D
 * geometry (flat-shaded faces, classic DX-tutorial topology).
 *
 * Environment knobs:
 *   DMN_FRAMES=N    auto-exit after N frames (CI).
 *
 * Pass --callback to use the callback-backed swapchain (embedder textures
 * presented through an MTKView) instead of the CAMetalLayer view backend.
 *   DMN_VALIDATE=1  read back the center pixel at frames 100 and 160:
 *                   both must show a saturated face color over the dark
 *                   clear, and the two colors must differ (the cube
 *                   actually spun). Prints "CUBE: PASS".
 *   DMN_DUMP=path   write path.<frame>.ppm at each validation frame.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <d3dcompiler.h>
#include <d3d11_1.h>
#include <dxgi1_6.h>

#include <windows.h>

#include "d3dmetal_native.h"
#include "cocoa_window.h"
#include "common/com.h"

static const bool g_validate = std::getenv("DMN_VALIDATE") != nullptr;
static const uint32_t g_maxFrames =
    std::getenv("DMN_FRAMES") ? (uint32_t)atoi(std::getenv("DMN_FRAMES")) : 0;

static const uint32_t kValidateFrameA = 100;
static const uint32_t kValidateFrameB = 160;

struct Vertex {
  float x, y, z;
  float r, g, b, a;
};

/* Row-major 4x4 for row-vector math: v' = v * M (matches the shader's
 * row_major declaration and the classic D3DX left-handed conventions). */
struct Mat4 {
  float m[16];
};

static Mat4 matMul(const Mat4& a, const Mat4& b) {
  Mat4 c = {};
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++)
      for (int j = 0; j < 4; j++)
        c.m[r * 4 + k] += a.m[r * 4 + j] * b.m[j * 4 + k];
  return c;
}

static Mat4 matRotX(float t) {
  float c = std::cos(t), s = std::sin(t);
  return {{1, 0, 0, 0,
           0, c, s, 0,
           0, -s, c, 0,
           0, 0, 0, 1}};
}

static Mat4 matRotY(float t) {
  float c = std::cos(t), s = std::sin(t);
  return {{c, 0, -s, 0,
           0, 1, 0, 0,
           s, 0, c, 0,
           0, 0, 0, 1}};
}

static Mat4 matTranslate(float x, float y, float z) {
  return {{1, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0,
           x, y, z, 1}};
}

static Mat4 matPerspectiveFovLH(float fovy, float aspect, float zn, float zf) {
  float ys = 1.0f / std::tan(fovy * 0.5f);
  float xs = ys / aspect;
  return {{xs, 0, 0, 0,
           0, ys, 0, 0,
           0, 0, zf / (zf - zn), 1,
           0, 0, -zn * zf / (zf - zn), 0}};
}

const std::string g_vertexShaderCode =
  "cbuffer vs_cb : register(b0) {\n"
  "  row_major float4x4 mvp;\n"
  "};\n"
  "struct vs_out {\n"
  "  float4 pos   : SV_POSITION;\n"
  "  float4 color : COLOR0;\n"
  "};\n"
  "vs_out main(float3 v_pos : POSITION, float4 v_color : COLOR) {\n"
  "  vs_out o;\n"
  "  o.pos   = mul(float4(v_pos, 1.0f), mvp);\n"
  "  o.color = v_color;\n"
  "  return o;\n"
  "}\n";

const std::string g_pixelShaderCode =
  "float4 main(float4 pos : SV_POSITION, float4 color : COLOR0) : SV_TARGET {\n"
  "  return color;\n"
  "}\n";

class CubeApp {

public:

  CubeApp(cocoa_window_t* cocoaWindow, HWND window)
  : m_cocoaWindow(cocoaWindow), m_window(window) {
    Com<ID3D11Device> device;

    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;

    HRESULT status = D3D11CreateDevice(
      nullptr, D3D_DRIVER_TYPE_HARDWARE,
      nullptr, 0, &fl, 1, D3D11_SDK_VERSION,
      &device, nullptr, nullptr);

    if (FAILED(status)) {
      std::cerr << "Failed to create D3D11 device" << std::endl;
      return;
    }

    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&m_device)))) {
      std::cerr << "Failed to query ID3D11Device1" << std::endl;
      return;
    }

    m_device->GetImmediateContext1(&m_context);

    Com<IDXGIDevice> dxgiDevice;
    Com<IDXGIAdapter> adapter;
    Com<IDXGIFactory2> factory;

    if (FAILED(m_device->QueryInterface(IID_PPV_ARGS(&dxgiDevice)))
     || FAILED(dxgiDevice->GetAdapter(&adapter))
     || FAILED(adapter->GetParent(IID_PPV_ARGS(&factory)))) {
      std::cerr << "Failed to query DXGI factory" << std::endl;
      return;
    }

    DXGI_SWAP_CHAIN_DESC1 swapDesc = {};
    swapDesc.Width       = m_windowSizeW;
    swapDesc.Height      = m_windowSizeH;
    swapDesc.Format      = DXGI_FORMAT_B8G8R8A8_UNORM;
    swapDesc.SampleDesc  = { 1, 0 };
    swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapDesc.BufferCount = 3;
    swapDesc.Scaling     = DXGI_SCALING_NONE;
    swapDesc.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapDesc.AlphaMode   = DXGI_ALPHA_MODE_UNSPECIFIED;

    if (FAILED(factory->CreateSwapChainForHwnd(m_device.ptr(), m_window,
        &swapDesc, nullptr, nullptr, &m_swapChain))) {
      std::cerr << "Failed to create DXGI swap chain" << std::endl;
      return;
    }

    Com<ID3DBlob> vsBlob;
    Com<ID3DBlob> psBlob;

    if (FAILED(D3DCompile(g_vertexShaderCode.data(), g_vertexShaderCode.size(),
        "Vertex shader", nullptr, nullptr, "main", "vs_5_0", 0, 0, &vsBlob, nullptr))) {
      std::cerr << "Failed to compile vertex shader" << std::endl;
      return;
    }

    if (FAILED(D3DCompile(g_pixelShaderCode.data(), g_pixelShaderCode.size(),
        "Pixel shader", nullptr, nullptr, "main", "ps_5_0", 0, 0, &psBlob, nullptr))) {
      std::cerr << "Failed to compile pixel shader" << std::endl;
      return;
    }

    if (FAILED(m_device->CreateVertexShader(vsBlob->GetBufferPointer(),
        vsBlob->GetBufferSize(), nullptr, &m_vs))
     || FAILED(m_device->CreatePixelShader(psBlob->GetBufferPointer(),
        psBlob->GetBufferSize(), nullptr, &m_ps))) {
      std::cerr << "Failed to create shaders" << std::endl;
      return;
    }

    std::array<D3D11_INPUT_ELEMENT_DESC, 2> layoutDesc = {{
      { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
      { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    }};

    if (FAILED(m_device->CreateInputLayout(layoutDesc.data(), layoutDesc.size(),
        vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_inputLayout))) {
      std::cerr << "Failed to create input layout" << std::endl;
      return;
    }

    /* Classic DX-tutorial cube: 8 corner positions, 36 indices grouped per
     * face, expanded into a flat 36-vertex buffer with one color per face
     * (CW winding for the default cull-back rasterizer state). */
    static const float corners[8][3] = {
      {-1,  1, -1}, { 1,  1, -1}, { 1,  1,  1}, {-1,  1,  1},
      {-1, -1, -1}, { 1, -1, -1}, { 1, -1,  1}, {-1, -1,  1},
    };
    static const uint32_t indices[36] = {
      3, 1, 0,  2, 1, 3,   /* top    (+Y) */
      0, 5, 4,  1, 5, 0,   /* front  (-Z) */
      3, 4, 7,  0, 4, 3,   /* left   (-X) */
      1, 6, 5,  2, 6, 1,   /* right  (+X) */
      2, 7, 6,  3, 7, 2,   /* back   (+Z) */
      6, 4, 5,  7, 4, 6,   /* bottom (-Y) */
    };
    static const float faceColors[6][4] = {
      {1, 0, 0, 1},  /* top:    red     */
      {0, 1, 0, 1},  /* front:  green   */
      {0, 0, 1, 1},  /* left:   blue    */
      {1, 1, 0, 1},  /* right:  yellow  */
      {0, 1, 1, 1},  /* back:   cyan    */
      {1, 0, 1, 1},  /* bottom: magenta */
    };

    std::array<Vertex, 36> vertexData;
    for (uint32_t i = 0; i < 36; i++) {
      const float* p = corners[indices[i]];
      const float* c = faceColors[i / 6];
      vertexData[i] = { p[0], p[1], p[2], c[0], c[1], c[2], c[3] };
    }

    D3D11_BUFFER_DESC vboDesc = {};
    vboDesc.ByteWidth = sizeof(vertexData);
    vboDesc.Usage     = D3D11_USAGE_IMMUTABLE;
    vboDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA vboData = {};
    vboData.pSysMem = vertexData.data();

    if (FAILED(m_device->CreateBuffer(&vboDesc, &vboData, &m_vbo))) {
      std::cerr << "Failed to create vertex buffer" << std::endl;
      return;
    }

    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth      = sizeof(Mat4);
    cbDesc.Usage          = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    if (FAILED(m_device->CreateBuffer(&cbDesc, nullptr, &m_cbVs))) {
      std::cerr << "Failed to create constant buffer" << std::endl;
      return;
    }

    m_initialized = true;
  }


  ~CubeApp() {
    if (m_context != nullptr)
      m_context->ClearState();
  }


  bool run() {
    if (!m_initialized)
      return false;

    if (m_occluded) {
      HRESULT hr = m_swapChain->Present(0, DXGI_PRESENT_TEST);
      m_occluded = hr == DXGI_STATUS_OCCLUDED;

      if (m_occluded)
        return true;
    }

    if (!beginFrame())
      return true;

    /* Deterministic tumble so validation frames are reproducible. */
    float t = float(m_frameTotal) * 0.02f;
    Mat4 world = matMul(matRotX(t * 0.7f), matRotY(t));
    Mat4 view  = matTranslate(0.0f, 0.0f, 4.0f);
    Mat4 proj  = matPerspectiveFovLH(
        3.14159265f / 3.0f,
        float(m_windowSizeW) / float(m_windowSizeH), 0.5f, 100.0f);
    Mat4 mvp = matMul(matMul(world, view), proj);

    D3D11_MAPPED_SUBRESOURCE sr = {};
    m_context->Map(m_cbVs.ptr(), 0, D3D11_MAP_WRITE_DISCARD, 0, &sr);
    memcpy(sr.pData, &mvp, sizeof(mvp));
    m_context->Unmap(m_cbVs.ptr(), 0);

    m_context->Draw(36, 0);

    if (!endFrame())
      return false;

    updateFps();
    return true;
  }


  uint32_t frameTotal() const {
    return m_frameTotal;
  }


  bool validated() const {
    return !g_validate || m_validatedB;
  }


  bool beginFrame() {
    uint32_t newWindowSizeW = m_windowSizeW;
    uint32_t newWindowSizeH = m_windowSizeH;
    cocoa_window_get_content_size(m_cocoaWindow, &newWindowSizeW, &newWindowSizeH);

    if (m_windowSizeW != newWindowSizeW || m_windowSizeH != newWindowSizeH) {
      m_context->ClearState();

      DXGI_SWAP_CHAIN_DESC1 desc;
      m_swapChain->GetDesc1(&desc);

      if (FAILED(m_swapChain->ResizeBuffers(desc.BufferCount,
          newWindowSizeW, newWindowSizeH, desc.Format, desc.Flags))) {
        std::cerr << "Failed to resize back buffers" << std::endl;
        return false;
      }

      m_windowSizeW = newWindowSizeW;
      m_windowSizeH = newWindowSizeH;
    }

    Com<ID3D11Texture2D> backBuffer;
    Com<ID3D11RenderTargetView> rtv;

    if (FAILED(m_swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) {
      std::cerr << "Failed to get swap chain back buffer" << std::endl;
      return false;
    }

    if (FAILED(m_device->CreateRenderTargetView(backBuffer.ptr(), nullptr, &rtv))) {
      std::cerr << "Failed to create render target view" << std::endl;
      return false;
    }

    D3D11_TEXTURE2D_DESC depthDesc = {};
    depthDesc.Width      = m_windowSizeW;
    depthDesc.Height     = m_windowSizeH;
    depthDesc.MipLevels  = 1;
    depthDesc.ArraySize  = 1;
    depthDesc.Format     = DXGI_FORMAT_D32_FLOAT;
    depthDesc.SampleDesc = { 1, 0 };
    depthDesc.Usage      = D3D11_USAGE_DEFAULT;
    depthDesc.BindFlags  = D3D11_BIND_DEPTH_STENCIL;

    Com<ID3D11Texture2D> depthTex;
    Com<ID3D11DepthStencilView> dsv;

    if (FAILED(m_device->CreateTexture2D(&depthDesc, nullptr, &depthTex))
     || FAILED(m_device->CreateDepthStencilView(depthTex.ptr(), nullptr, &dsv))) {
      std::cerr << "Failed to create depth buffer" << std::endl;
      return false;
    }

    FLOAT clearColor[4] = { 0.10f, 0.10f, 0.15f, 1.0f };
    m_context->OMSetRenderTargets(1, &rtv, dsv.ptr());
    m_context->ClearRenderTargetView(rtv.ptr(), clearColor);
    m_context->ClearDepthStencilView(dsv.ptr(), D3D11_CLEAR_DEPTH, 1.0f, 0);

    m_context->VSSetShader(m_vs.ptr(), nullptr, 0);
    m_context->PSSetShader(m_ps.ptr(), nullptr, 0);

    m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_context->IASetInputLayout(m_inputLayout.ptr());

    D3D11_VIEWPORT viewport = {};
    viewport.Width    = float(m_windowSizeW);
    viewport.Height   = float(m_windowSizeH);
    viewport.MaxDepth = 1.0f;
    m_context->RSSetViewports(1, &viewport);

    uint32_t vsStride = sizeof(Vertex);
    uint32_t vsOffset = 0;
    m_context->IASetVertexBuffers(0, 1, &m_vbo, &vsStride, &vsOffset);
    m_context->VSSetConstantBuffers(0, 1, &m_cbVs);
    return true;
  }


  bool endFrame() {
    m_frameTotal++;

    if (g_validate
     && (m_frameTotal == kValidateFrameA || m_frameTotal == kValidateFrameB)
     && !validateFrame())
      return false;

    HRESULT hr = m_swapChain->Present(1, 0);
    m_occluded = hr == DXGI_STATUS_OCCLUDED;
    return true;
  }


  /* Reads the backbuffer before Present. The cube (half-extent 1 at
   * distance 4, 60 degree fov) always covers the window center, so the
   * center pixel must be a saturated face color over the dark clear; the
   * face seen at frame 100 must differ from the one at frame 160. */
  bool validateFrame() {
    Com<ID3D11Texture2D> backBuffer;
    if (FAILED(m_swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) {
      std::cerr << "VALIDATE: FAILED to get back buffer" << std::endl;
      return false;
    }

    D3D11_TEXTURE2D_DESC desc;
    backBuffer->GetDesc(&desc);
    desc.Usage          = D3D11_USAGE_STAGING;
    desc.BindFlags      = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags      = 0;

    Com<ID3D11Texture2D> staging;
    if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &staging))) {
      std::cerr << "VALIDATE: FAILED to create staging texture" << std::endl;
      return false;
    }

    m_context->CopyResource(staging.ptr(), backBuffer.ptr());

    D3D11_MAPPED_SUBRESOURCE sr = {};
    if (FAILED(m_context->Map(staging.ptr(), 0, D3D11_MAP_READ, 0, &sr))) {
      std::cerr << "VALIDATE: FAILED to map staging texture" << std::endl;
      return false;
    }

    auto pixelAt = [&](uint32_t x, uint32_t y) -> uint32_t {
      uint32_t p;
      memcpy(&p, (const uint8_t*)sr.pData + y * sr.RowPitch + x * 4, 4);
      return p; /* BGRA8: B=byte0 G=byte1 R=byte2 */
    };

    uint32_t center = pixelAt(m_windowSizeW / 2, m_windowSizeH / 2);
    uint32_t corner = pixelAt(10, 10);

    if (const char* dump = std::getenv("DMN_DUMP")) {
      char path[512];
      snprintf(path, sizeof(path), "%s.%u.ppm", dump, m_frameTotal);
      if (FILE* f = fopen(path, "wb")) {
        fprintf(f, "P6\n%u %u\n255\n", m_windowSizeW, m_windowSizeH);
        for (uint32_t y = 0; y < m_windowSizeH; y++) {
          for (uint32_t x = 0; x < m_windowSizeW; x++) {
            uint32_t p = pixelAt(x, y);
            uint8_t rgb[3] = { (uint8_t)(p >> 16), (uint8_t)(p >> 8),
                               (uint8_t)p };
            fwrite(rgb, 1, 3, f);
          }
        }
        fclose(f);
        printf("CUBE: frame %u dumped to %s\n", m_frameTotal, path);
      }
    }

    m_context->Unmap(staging.ptr(), 0);

    uint8_t r = center >> 16, g = center >> 8, b = center;
    uint8_t maxc = std::max({r, g, b}), minc = std::min({r, g, b});

    /* Clear is (0.10, 0.10, 0.15) -> roughly (26, 26, 38). */
    uint8_t cr = corner >> 16, cg = corner >> 8, cb = corner;
    bool cornerOk = cr < 60 && cg < 60 && cb < 60;
    bool centerOk = maxc > 200 && (maxc - minc) > 120;

    printf("CUBE: frame %u center=(%u,%u,%u) corner=(%u,%u,%u) %s%s\n",
           m_frameTotal, r, g, b, cr, cg, cb,
           centerOk ? "" : "[center not a face color] ",
           cornerOk ? "" : "[corner not clear color]");

    if (!centerOk || !cornerOk)
      return false;

    if (m_frameTotal == kValidateFrameA) {
      m_colorA = center;
      return true;
    }

    /* Frame B: the visible face must have changed. */
    uint8_t ar = m_colorA >> 16, ag = m_colorA >> 8, ab = m_colorA;
    uint32_t dist = std::abs((int)ar - r) + std::abs((int)ag - g) +
                    std::abs((int)ab - b);
    bool spun = dist > 64;
    printf("CUBE: face color delta |frame%u - frame%u| = %u -> %s\n",
           kValidateFrameA, kValidateFrameB, dist,
           spun ? "spinning" : "NOT spinning");
    if (!spun)
      return false;

    m_validatedB = true;
    printf("CUBE: PASS\n");
    return true;
  }


  void updateFps() {
    using namespace std::chrono;

    if (m_lastUpdate.time_since_epoch().count() == 0)
      m_lastUpdate = steady_clock::now();

    auto now = steady_clock::now();

    m_frameCount++;

    if (now - m_lastUpdate < seconds(1))
      return;

    double fps = double(m_frameCount)
               / duration<double>(now - m_lastUpdate).count();

    char title[128];
    snprintf(title, sizeof(title), "D3D11 cube (%.1f FPS)", fps);
    cocoa_window_set_title(m_cocoaWindow, title);

    m_lastUpdate = now;
    m_frameCount = 0;
  }

private:

  cocoa_window_t*               m_cocoaWindow;
  HWND                          m_window;
  uint32_t                      m_windowSizeW = 800;
  uint32_t                      m_windowSizeH = 600;
  bool                          m_initialized = false;
  bool                          m_occluded = false;
  bool                          m_validatedB = false;
  uint32_t                      m_colorA = 0;

  Com<ID3D11Device1>            m_device;
  Com<ID3D11DeviceContext1>     m_context;
  Com<IDXGISwapChain1>          m_swapChain;

  Com<ID3D11Buffer>             m_vbo;
  Com<ID3D11Buffer>             m_cbVs;
  Com<ID3D11InputLayout>        m_inputLayout;

  Com<ID3D11VertexShader>       m_vs;
  Com<ID3D11PixelShader>        m_ps;

  std::chrono::steady_clock::time_point m_lastUpdate = { };

  uint32_t                      m_frameCount = 0;
  uint32_t                      m_frameTotal = 0;

};

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  if (!cocoa_app_init()) {
    std::cerr << "Failed to initialize Cocoa application" << std::endl;
    return 1;
  }

  if (dmn_init(nullptr) != DMN_SUCCESS) {
    std::cerr << "Failed to initialize d3dmetal-native" << std::endl;
    return 1;
  }

  cocoa_window_t* window = cocoa_window_create("D3D11 cube", 800, 600, true);
  if (!window) {
    std::cerr << "Failed to create window" << std::endl;
    return 1;
  }

  dmn_window_t dmnWindow =
      cocoa_window_create_dmn(window, cocoa_arg_callback(argc, argv));
  if (!dmnWindow) {
    std::cerr << "Failed to create dmn window" << std::endl;
    return 1;
  }

  HWND hwnd = (HWND)dmn_window_get_hwnd(dmnWindow);

  int ret = 0;

  {
    CubeApp app(window, hwnd);

    while (cocoa_app_poll()) {
      if (!app.run()) {
        ret = 1;
        break;
      }
      if (g_maxFrames && app.frameTotal() >= g_maxFrames)
        break;
    }

    if (!app.validated())
      ret = 1;
  }

  dmn_window_destroy(dmnWindow);
  cocoa_window_destroy(window);
  cocoa_app_shutdown();
  return ret;
}
