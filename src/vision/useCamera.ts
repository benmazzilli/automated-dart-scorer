import { useCallback, useEffect, useRef, useState } from 'react'
import { toGrayscale, type GrayImage } from './frameDiff'

export type CameraStatus = 'idle' | 'starting' | 'live' | 'denied' | 'unavailable' | 'error'

export interface CameraState {
  status: CameraStatus
  message: string | null
}

export interface UseCameraOptions {
  width: number
  height: number
  /** Frames per second handed to `onFrame`. */
  fps: number
  onFrame: (frame: GrayImage) => void
  enabled: boolean
}

/**
 * Camera capture, sampled down to a working resolution and delivered as
 * greyscale frames.
 *
 * Two iOS details are load-bearing. The video element must carry `playsInline`
 * or Safari takes the stream fullscreen, and permission is not persisted for
 * installed PWAs — it is re-asked on every launch, which is a WebKit
 * behaviour, not something the page can avoid.
 */
/**
 * Draw the video into the canvas cropped exactly the way CSS `object-cover`
 * crops it on screen.
 *
 * This has to match, and it is not a detail. A phone hands over whatever
 * aspect ratio it feels like — often 16:9 — while the preview is a fixed 4:3
 * box, so the picture on screen is centre-cropped. Capturing the *whole* frame
 * instead would mean a tap at the middle of the preview does not correspond to
 * the middle of the captured frame, and calibration taps would land on the
 * wrong part of the board. The scores that follow are then wrong in a way that
 * looks like a broken detector rather than a coordinate mismatch.
 */
function drawCovering(
  context: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  width: number,
  height: number,
): void {
  const sourceWidth = video.videoWidth
  const sourceHeight = video.videoHeight
  if (sourceWidth === 0 || sourceHeight === 0) return

  const scale = Math.max(width / sourceWidth, height / sourceHeight)
  const cropWidth = width / scale
  const cropHeight = height / scale
  const cropX = (sourceWidth - cropWidth) / 2
  const cropY = (sourceHeight - cropHeight) / 2

  context.drawImage(video, cropX, cropY, cropWidth, cropHeight, 0, 0, width, height)
}

export function useCamera({ width, height, fps, onFrame, enabled }: UseCameraOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<number | null>(null)
  const onFrameRef = useRef(onFrame)
  const [state, setState] = useState<CameraState>({ status: 'idle', message: null })

  // Kept in a ref so changing the callback does not tear the stream down.
  useEffect(() => {
    onFrameRef.current = onFrame
  }, [onFrame])

  const stop = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setState({ status: 'idle', message: null })
  }, [])

  useEffect(() => {
    if (!enabled) {
      stop()
      return
    }

    let cancelled = false

    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setState({
          status: 'unavailable',
          message: 'This browser will not give the page a camera. Safari or Chrome over HTTPS will.',
        })
        return
      }

      setState({ status: 'starting', message: null })

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }

        streamRef.current = stream
        const video = videoRef.current
        if (!video) return
        video.srcObject = stream
        await video.play()
        if (cancelled) return

        setState({ status: 'live', message: null })

        const canvas = canvasRef.current ?? document.createElement('canvas')
        canvasRef.current = canvas
        canvas.width = width
        canvas.height = height
        const context = canvas.getContext('2d', { willReadFrequently: true })
        if (!context) {
          setState({ status: 'error', message: 'Could not read frames from the camera.' })
          return
        }

        timerRef.current = window.setInterval(() => {
          const source = videoRef.current
          if (!source || source.readyState < 2) return
          drawCovering(context, source, width, height)
          const { data } = context.getImageData(0, 0, width, height)
          onFrameRef.current(toGrayscale(data, width, height))
        }, Math.round(1000 / fps))
      } catch (error) {
        if (cancelled) return
        const name = error instanceof DOMException ? error.name : ''
        if (name === 'NotAllowedError' || name === 'SecurityError') {
          setState({
            status: 'denied',
            message:
              'Camera access was refused. Allow it in Settings, then reopen. Installed web apps ask again each launch.',
          })
        } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
          setState({ status: 'unavailable', message: 'No camera was found on this device.' })
        } else {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'The camera failed to start.',
          })
        }
      }
    }

    void start()
    return () => {
      cancelled = true
      stop()
    }
  }, [enabled, width, height, fps, stop])

  return { videoRef, state, stop }
}
