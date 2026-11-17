/* Full-surface LCD compositing. Mask values match the reference clock's CSS.
   This layer never reads page pixels, text, media, or the embedded chat. */
(() => {
  "use strict";
  const canvas = document.createElement("canvas");
  canvas.className = "lcd-render-surface";
  canvas.setAttribute("aria-hidden", "true");
  document.body.append(canvas);
  let gl;
  let program;

  function fallback() {
    document.body.classList.add("lcd-render-fallback");
    canvas.hidden = true;
    canvas.dataset.renderer = "css";
  }

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error("LCD mask unavailable");
    return shader;
  }

  function render() {
    if (!gl || !program || gl.isContextLost()) return;
    const width = window.innerWidth;
    const height = window.innerHeight;
    // A static mask needs no animation loop and no sampled DOM texture.
    const density = Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(12_000_000 / Math.max(1, width * height)));
    canvas.width = Math.max(1, Math.round(width * density));
    canvas.height = Math.max(1, Math.round(height * density));
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.uniform2f(gl.getUniformLocation(program, "u_scale"), canvas.width / width, canvas.height / height);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function initialize() {
    try {
      gl = canvas.getContext("webgl", { alpha: true, antialias: false, depth: false, stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
      if (!gl) return fallback();
      const vertex = compile(gl.VERTEX_SHADER, `
        attribute vec2 a_position;
        void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
      `);
      const fragment = compile(gl.FRAGMENT_SHADER, `
        precision highp float;
        uniform vec2 u_scale;
        void main() {
          vec2 pixel = gl_FragCoord.xy / u_scale;
          float scanline = 1.0 - step(1.0, mod(pixel.y, 3.0));
          float column = 1.0 - step(1.0, mod(pixel.x, 3.0));
          float mask = 1.0 - (1.0 - 0.38 * scanline) * (1.0 - 0.22 * column);
          gl_FragColor = vec4(0.0, 0.0, 0.0, mask);
        }
      `);
      program = gl.createProgram();
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.linkProgram(program);
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return fallback();
      gl.useProgram(program);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
      const position = gl.getAttribLocation(program, "a_position");
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      canvas.hidden = false;
      canvas.dataset.renderer = "webgl";
      document.body.classList.remove("lcd-render-fallback");
      render();
    } catch (_error) {
      fallback();
    }
  }

  canvas.addEventListener("webglcontextlost", event => { event.preventDefault(); fallback(); });
  canvas.addEventListener("webglcontextrestored", initialize);
  window.addEventListener("resize", render);
  window.visualViewport?.addEventListener("resize", render);
  initialize();
})();
