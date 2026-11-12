import { useLayoutEffect, useRef, useState } from 'react'

/** The rendered width of the element `ref` points at, kept current on resize. Lets an SVG use
 *  its real pixel width as its viewBox, so its text keeps a fixed size on every screen. */
export function useWidth<T extends HTMLElement>(initial = 640) {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(initial)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => setWidth(Math.round(el.clientWidth))
    measure()
    const watch = new ResizeObserver(measure)
    watch.observe(el)
    return () => watch.disconnect()
  })
  return [ref, width] as const
}
