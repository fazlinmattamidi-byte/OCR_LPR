let ortModuleCache: Promise<any> | null = null;

const ORT_PUBLIC_PATH = '/ort-wasm/';
const ORT_WASM_MODULE = `${ORT_PUBLIC_PATH}ort-wasm-simd-threaded.mjs`;
const ORT_WASM_BINARY = `${ORT_PUBLIC_PATH}ort-wasm-simd-threaded.wasm`;
const ORT_JSEP_MODULE = `${ORT_PUBLIC_PATH}ort-wasm-simd-threaded.jsep.mjs`;
const ORT_JSEP_BINARY = `${ORT_PUBLIC_PATH}ort-wasm-simd-threaded.jsep.wasm`;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existingScript = document.querySelector<HTMLScriptElement>(`script[src="${src}"]`);
    if (existingScript) {
      if ((window as any).ort) {
        resolve();
        return;
      }

      existingScript.addEventListener('load', () => resolve(), { once: true });
      existingScript.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });
}

/**
 * Load ONNX Runtime from the installed npm package instead of a CDN script.
 * This keeps local scanning functional on restricted networks and mobile devices.
 */
export async function getOrt(): Promise<any> {
  if (typeof window === 'undefined') return null;

  const existingOrt = (window as any).ort;
  if (existingOrt) return existingOrt;

  if (!ortModuleCache) {
    ortModuleCache = loadScript('/ort-wasm/ort.min.js').then(() => {
      const ort = (window as any).ort;
      if (!ort) throw new Error('ONNX Runtime loaded but did not expose window.ort');
      return ort;
    });
  }

  return ortModuleCache;
}

export function configureOrtWasm(ort: any, useWebGpuLoader = false): void {
  if (!ort?.env?.wasm) return;

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = {
    mjs: useWebGpuLoader ? ORT_JSEP_MODULE : ORT_WASM_MODULE,
    wasm: useWebGpuLoader ? ORT_JSEP_BINARY : ORT_WASM_BINARY,
  };
}
