/**
 * Frame differencing and the morphology needed to clean the result up.
 *
 * Detecting a dart by looking for "a dart" in a still image is the hard
 * version of this problem, and it is what the previous implementation tried
 * to do with hand-tuned colour thresholds. Detecting *what changed* between
 * the board a moment ago and the board now is a far easier question, and the
 * answer is nearly always exactly one thing: the dart that just landed.
 */

/** A single-channel image. */
export interface GrayImage {
  data: Uint8Array
  width: number
  height: number
}

/** A binary mask, 0 or 1 per pixel. */
export interface Mask {
  data: Uint8Array
  width: number
  height: number
  /** Number of set pixels, kept so callers do not have to re-count. */
  count: number
}

/** Convert RGBA pixels, as produced by a canvas, to greyscale. */
export function toGrayscale(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): GrayImage {
  const data = new Uint8Array(width * height)
  for (let i = 0, p = 0; i < data.length; i++, p += 4) {
    // Rec. 601 luma. Integer weights keep this fast on a phone.
    data[i] = (rgba[p]! * 77 + rgba[p + 1]! * 150 + rgba[p + 2]! * 29) >> 8
  }
  return { data, width, height }
}

/** Absolute difference of two images, thresholded into a binary mask. */
export function differenceMask(a: GrayImage, b: GrayImage, threshold: number): Mask {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error('cannot difference images of different sizes')
  }
  const data = new Uint8Array(a.data.length)
  let count = 0
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(a.data[i]! - b.data[i]!) >= threshold) {
      data[i] = 1
      count++
    }
  }
  return { data, width: a.width, height: a.height, count }
}

/** Fraction of pixels that differ — the motion signal. */
export function changedFraction(mask: Mask): number {
  return mask.count / (mask.width * mask.height)
}

/**
 * Morphological erosion with a 3×3 square. Removes single-pixel speckle.
 */
export function erode(mask: Mask): Mask {
  return applyMorphology(mask, true)
}

/**
 * Morphological dilation with a 3×3 square. Fills the gaps erosion leaves and
 * reconnects a dart split by a highlight along the barrel.
 */
export function dilate(mask: Mask): Mask {
  return applyMorphology(mask, false)
}

function applyMorphology(mask: Mask, isErosion: boolean): Mask {
  const { width, height, data } = mask
  const out = new Uint8Array(data.length)
  let count = 0

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let hit = isErosion ? 1 : 0
      for (let dy = -1; dy <= 1 && (isErosion ? hit : !hit); dy++) {
        const ny = y + dy
        if (ny < 0 || ny >= height) {
          // Treat outside the image as empty, so erosion trims the border.
          if (isErosion) hit = 0
          continue
        }
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx
          if (nx < 0 || nx >= width) {
            if (isErosion) hit = 0
            continue
          }
          const value = data[ny * width + nx]!
          if (isErosion && value === 0) {
            hit = 0
            break
          }
          if (!isErosion && value === 1) {
            hit = 1
            break
          }
        }
      }
      out[y * width + x] = hit
      if (hit) count++
    }
  }

  return { data: out, width, height, count }
}

/** Erosion then dilation: removes speckle while keeping the dart's size. */
export function open(mask: Mask): Mask {
  return dilate(erode(mask))
}

/** Dilation then erosion: closes small holes and joins near-touching parts. */
export function close(mask: Mask): Mask {
  return erode(dilate(mask))
}

/** Restrict a mask to pixels satisfying a predicate, e.g. inside the board. */
export function maskWhere(mask: Mask, keep: (x: number, y: number) => boolean): Mask {
  const out = new Uint8Array(mask.data.length)
  let count = 0
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      const index = y * mask.width + x
      if (mask.data[index] === 1 && keep(x, y)) {
        out[index] = 1
        count++
      }
    }
  }
  return { data: out, width: mask.width, height: mask.height, count }
}
