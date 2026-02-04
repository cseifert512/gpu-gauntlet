/**
 * WebGPU device management.
 *
 * Handles adapter/device acquisition, capability detection,
 * and graceful degradation when WebGPU is unavailable.
 */

export interface GPUContext {
  device: GPUDevice;
  adapter: GPUAdapter;
  limits: GPUSupportedLimits;
}

let cachedContext: GPUContext | null = null;

/**
 * Check if WebGPU is available in current browser.
 */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * Initialize WebGPU device.
 *
 * Returns cached context if already initialized.
 * Returns null if WebGPU is unavailable.
 *
 * @returns GPU context or null if unavailable
 */
export async function initGPU(): Promise<GPUContext | null> {
  if (cachedContext) return cachedContext;

  if (!isWebGPUAvailable()) {
    console.warn('[GPU] WebGPU not available');
    return null;
  }

  try {
    console.log('[GPU] Requesting adapter...');
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });

    if (!adapter) {
      console.warn('[GPU] No WebGPU adapter found');
      return null;
    }

    // Log adapter limits for debugging
    console.log('[GPU] Adapter limits:', {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    });

    // Request device with limits capped to what adapter supports
    console.log('[GPU] Requesting device...');
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          256 * 1024 * 1024,
          adapter.limits.maxStorageBufferBindingSize
        ),
        maxComputeWorkgroupsPerDimension: Math.min(
          65535,
          adapter.limits.maxComputeWorkgroupsPerDimension
        ),
        maxComputeInvocationsPerWorkgroup: Math.min(
          256,
          adapter.limits.maxComputeInvocationsPerWorkgroup
        ),
        maxComputeWorkgroupSizeX: Math.min(
          256,
          adapter.limits.maxComputeWorkgroupSizeX
        ),
        maxComputeWorkgroupSizeY: Math.min(
          256,
          adapter.limits.maxComputeWorkgroupSizeY
        ),
        maxComputeWorkgroupSizeZ: Math.min(
          64,
          adapter.limits.maxComputeWorkgroupSizeZ
        ),
      },
    });

    console.log('[GPU] Device acquired successfully');

    // Handle device loss
    device.lost.then((info) => {
      console.error('[GPU] Device lost:', info.message);
      cachedContext = null;
    });

    // Error handling
    device.onuncapturederror = (event) => {
      console.error('[GPU] Uncaptured error:', event.error);
    };

    cachedContext = { device, adapter, limits: device.limits };
    return cachedContext;
  } catch (e) {
    console.error('[GPU] Initialization failed:', e);
    return null;
  }
}

/**
 * Get cached GPU context.
 * Returns null if not initialized.
 */
export function getGPUContext(): GPUContext | null {
  return cachedContext;
}

/**
 * Destroy GPU context (for testing/cleanup).
 */
export function destroyGPU(): void {
  if (cachedContext) {
    cachedContext.device.destroy();
    cachedContext = null;
  }
}

/**
 * Reset cached context (for testing).
 */
export function resetGPUCache(): void {
  cachedContext = null;
}

