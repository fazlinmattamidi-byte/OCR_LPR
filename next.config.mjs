import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Auto-copy ONNX Runtime browser assets from node_modules to public/ort-wasm.
// This removes the CDN dependency which can be slow or blocked on mobile.
function copyOnnxWasm() {
  const src = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
  const dest = path.join(__dirname, 'public', 'ort-wasm');
  if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
  for (const file of fs.readdirSync(src)) {
    const isWasmLoader = file.startsWith('ort-wasm-simd-threaded') && (file.endsWith('.wasm') || file.endsWith('.mjs'));
    const isBrowserRuntime = file === 'ort.min.js';

    if (isWasmLoader || isBrowserRuntime) {
      const destFile = path.join(dest, file);
      // Only copy if source is newer (avoid unnecessary I/O on hot-reload)
      const srcStat = fs.statSync(path.join(src, file));
      const destStat = fs.existsSync(destFile) ? fs.statSync(destFile) : null;
      if (!destStat || srcStat.mtimeMs > destStat.mtimeMs) {
        fs.copyFileSync(path.join(src, file), destFile);
      }
    }
  }
}
copyOnnxWasm();

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  experimental: {
    esmExternals: 'loose',
  },
  webpack: (config, { isServer }) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };

    if (isServer) {
      config.externals = [...(config.externals || []), 'onnxruntime-web'];
    }

    return config;
  },
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'credentialless',
          },
          {
            key: 'Cache-Control',
            value: 'no-cache, no-store, must-revalidate',
          },
        ],
      },
    ];
  },
};

export default nextConfig;
