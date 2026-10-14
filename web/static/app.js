const PYTHON_KEYWORDS = new Set([
  'import', 'from', 'as', 'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while',
  'in', 'not', 'and', 'or', 'with', 'try', 'except', 'finally', 'raise', 'yield', 'lambda',
  'None', 'True', 'False', 'pass', 'break', 'continue', 'global', 'nonlocal', 'assert', 'del',
  'is', 'async', 'await'
]);

const HOME = 'home';

const dom = {
  brand: document.getElementById('brand'),
  nav: document.getElementById('nav'),
  main: document.getElementById('main')
};

const app = {
  labs: [],
  subjects: [],
  view: HOME,
  collapsed: new Set(),
  viewers: []
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function pad(number) {
  return String(number).padStart(2, '0');
}

function pluralise(count, one, few, many) {
  const lastDigit = count % 10;
  const lastTwo = count % 100;
  if (lastDigit === 1 && lastTwo !== 11) return `${count} ${one}`;
  if (lastDigit >= 2 && lastDigit <= 4 && (lastTwo < 12 || lastTwo > 14)) return `${count} ${few}`;
  return `${count} ${many}`;
}

function formatValue(control, value) {
  const digits = Number(control.step) < 1 ? 2 : 0;
  return value.toFixed(digits) + (control.unit || '');
}

function escapeHtml(text) {
  return text.replace(/[&<>]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[character]));
}

function highlightPython(source) {
  const token = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_\u0400-\u04FF][A-Za-z0-9_\u0400-\u04FF]*)/g;
  let html = '';
  let cursor = 0;
  let match;

  while ((match = token.exec(source)) !== null) {
    html += escapeHtml(source.slice(cursor, match.index));
    cursor = token.lastIndex;

    const [, comment, string, number, word] = match;
    if (comment) html += `<span class="c-comment">${escapeHtml(comment)}</span>`;
    else if (string) html += `<span class="c-str">${escapeHtml(string)}</span>`;
    else if (number) html += `<span class="c-num">${number}</span>`;
    else if (PYTHON_KEYWORDS.has(word)) html += `<span class="c-kw">${word}</span>`;
    else if (source[token.lastIndex] === '(') html += `<span class="c-fn">${word}</span>`;
    else html += escapeHtml(word);
  }

  return html + escapeHtml(source.slice(cursor));
}

function groupByCourse(labs) {
  const subjects = new Map();

  for (const lab of labs) {
    if (!subjects.has(lab.course)) {
      subjects.set(lab.course, { id: lab.course, title: lab.courseTitle || lab.course, items: [] });
    }
    subjects.get(lab.course).items.push(lab);
  }

  return [...subjects.values()].map(subject => {
    subject.items.sort((a, b) => a.number - b.number);
    const numbers = subject.items.map(lab => lab.number);
    const first = pad(Math.min(...numbers));
    const last = pad(Math.max(...numbers));
    subject.sign = first === last ? first : `${first}—${last}`;
    subject.meta = pluralise(subject.items.length, 'робота', 'роботи', 'робіт');
    return subject;
  });
}

function subjectOf(lab) {
  return app.subjects.find(subject => subject.items.includes(lab));
}

const RECONNECT_MIN = 1000;
const RECONNECT_MAX = 15000;

class FrameStream {
  constructor(labId, demoId, handlers) {
    this.labId = labId;
    this.demoId = demoId;
    this.handlers = handlers;
    this.socket = null;
    this.attempt = 0;
    this.retry = null;
    this.stopped = false;
  }

  open() {
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const path = `/api/labs/${encodeURIComponent(this.labId)}/demos/${encodeURIComponent(this.demoId)}/stream`;
    const socket = new WebSocket(`${scheme}//${location.host}${path}`);
    socket.binaryType = 'blob';
    this.socket = socket;

    socket.addEventListener('open', () => {
      this.attempt = 0;
      this.handlers.onOpen?.();
    });

    socket.addEventListener('message', event => {
      if (typeof event.data !== 'string') {
        this.handlers.onFrame?.(event.data);
        return;
      }
      const message = JSON.parse(event.data);
      if (message.type === 'error') this.handlers.onError?.(message.detail);
      else if (message.type === 'readout') this.handlers.onReadout?.(message.rows);
      else if (message.type === 'hello') this.handlers.onHello?.();
    });

    socket.addEventListener('close', () => {
      if (this.stopped || this.socket !== socket) return;
      const delay = Math.min(RECONNECT_MIN * 2 ** this.attempt, RECONNECT_MAX);
      this.attempt += 1;
      this.handlers.onReconnecting?.(delay);
      this.retry = setTimeout(() => this.open(), delay);
    });
  }

  send(message) {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(message));
  }

  close() {
    this.stopped = true;
    clearTimeout(this.retry);
    this.socket?.close();
    this.socket = null;
  }
}

class Histogram {
  constructor(canvas, data) {
    this.canvas = canvas;
    this.data = data;
  }

  draw(center, width) {
    const context = this.canvas.getContext('2d');
    const { width: canvasWidth, height } = this.canvas;
    const { bins, min, max } = this.data;
    const peak = Math.max(...bins) || 1;

    context.clearRect(0, 0, canvasWidth, height);

    const barWidth = canvasWidth / bins.length;
    context.fillStyle = '#2B343A';
    bins.forEach((count, index) => {
      const barHeight = (Math.log1p(count) / Math.log1p(peak)) * (height - 8);
      context.fillRect(index * barWidth, height - barHeight, Math.max(barWidth - 1, 1), barHeight);
    });

    if (!Number.isFinite(center) || !Number.isFinite(width)) return;

    context.strokeStyle = '#5FD3D8';
    context.lineWidth = 1.5;
    for (const bound of [center - width / 2, center + width / 2]) {
      const x = ((bound - min) / (max - min)) * canvasWidth;
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
    }
  }
}

class DemoViewer {
  constructor(lab, demo) {
    this.demo = demo;
    this.params = Object.fromEntries(
      (demo.controls || []).map(control => [control.id, Number(control.value)]));

    this.image = el('img');
    this.image.alt = demo.title;
    this.rate = el('div', 'stage-badge stage-rate');
    this.error = el('div', 'error');

    this.stage = el('div', 'stage');
    this.stage.dataset.state = 'loading';
    this.status = el('div', 'stage-loader-text', 'ЗʼЄДНАННЯ…');
    this.loader = el('div', 'stage-loader');
    this.loader.append(el('div', 'stage-spinner'), this.status);
    this.image.addEventListener('load', () => { this.stage.dataset.state = 'ready'; }, { once: true });
    this.histogram = null;
    this.readout = null;
    this.presetButtons = [];
    this.fields = new Map();

    this.playing = false;
    this.objectUrl = null;
    this.frames = 0;
    this.measuredFrom = performance.now();

    this.stream = new FrameStream(lab.id, demo.id, {
      onOpen: () => this.resume(),
      onHello: () => this.waitFor('РЕНДЕР КАДРУ…'),
      onReconnecting: () => { this.rate.textContent = 'ЗʼЄДНАННЯ…'; this.waitFor('ЗʼЄДНАННЯ…'); },
      onFrame: blob => this.paint(blob),
      onReadout: rows => this.showReadout(rows),
      onError: detail => this.fail(detail)
    });
    this.onResize = () => this.sendSize();
  }

  start() {
    this.stream.open();
    window.addEventListener('resize', this.onResize);
    this.redrawHistogram();
  }

  destroy() {
    window.removeEventListener('resize', this.onResize);
    this.stream.close();
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
  }

  resume() {
    this.error.textContent = '';
    this.rate.textContent = '';
    this.sendSize();
    this.stream.send({ params: this.params, playing: this.playing });
  }

  waitFor(status) {
    if (this.stage.dataset.state === 'ready') return;
    this.stage.dataset.state = 'loading';
    this.status.textContent = status;
  }

  fail(detail) {
    this.error.textContent = detail;
    if (this.stage.dataset.state !== 'ready') {
      this.stage.dataset.state = 'failed';
      this.status.textContent = 'СЦЕНА НЕ ВІДПОВІЛА';
    }
  }

  sendSize() {
    const box = this.image.parentElement.getBoundingClientRect();
    const pixels = Math.min(Math.max(box.width, 128), 1024) * (window.devicePixelRatio || 1);
    const size = Math.round(pixels);
    this.stream.send({ width: size, height: size });
  }

  setParam(id, value) {
    this.params[id] = value;
    this.stream.send({ params: { [id]: value } });
    this.redrawHistogram();
    this.markActivePreset();
  }

  applyPreset(preset) {
    for (const [id, value] of Object.entries(preset.params)) {
      const field = this.fields.get(id);
      if (!field) continue;
      field.input.value = value;
      field.readout.textContent = formatValue(field.control, value);
      this.params[id] = value;
    }
    this.stream.send({ params: preset.params });
    this.redrawHistogram();
    this.markActivePreset();
  }

  markActivePreset() {
    for (const { button, preset } of this.presetButtons) {
      const active = Object.entries(preset.params)
        .every(([id, value]) => Number(this.params[id]) === Number(value));
      button.setAttribute('aria-pressed', String(active));
    }
  }

  togglePlay() {
    this.playing = !this.playing;
    this.stream.send({ playing: this.playing });
    if (!this.playing) this.rate.textContent = '';
    this.frames = 0;
    this.measuredFrom = performance.now();
    return this.playing;
  }

  paint(blob) {
    const url = URL.createObjectURL(blob);
    this.image.src = url;
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = url;
    this.countFrame();
  }

  countFrame() {
    this.frames += 1;
    const seconds = (performance.now() - this.measuredFrom) / 1000;
    if (seconds < 1) return;

    const rate = this.frames / seconds;
    this.rate.textContent = this.playing && rate >= 1 ? `${rate.toFixed(0)} КАДР/С` : '';
    this.frames = 0;
    this.measuredFrom = performance.now();
  }

  redrawHistogram() {
    this.histogram?.draw(Number(this.params.windowCenter), Number(this.params.windowWidth));
  }

  showReadout(rows) {
    if (!this.readout) return;
    this.readout.replaceChildren(...rows.map(row => {
      const line = el('div', 'readout-row');
      line.append(el('span', 'readout-label', row.label), el('span', 'readout-value', row.value));
      return line;
    }));
  }
}

function buildDemo(lab, demo) {
  const viewer = new DemoViewer(lab, demo);
  const layout = el('div', 'demo');
  const left = el('div', 'demo-left');
  const right = el('div', 'demo-right');

  viewer.stage.append(viewer.image, viewer.loader,
                      el('div', 'stage-badge', demo.title.toUpperCase()), viewer.rate);
  left.append(viewer.stage, buildControls(viewer, demo));

  if (demo.presets) left.append(buildPresets(viewer, demo.presets));
  if (demo.animate) left.append(buildPlayButton(viewer));

  right.append(...buildPanels(viewer, demo), viewer.error);

  layout.append(left, right);
  app.viewers.push(viewer);
  viewer.start();
  return layout;
}

function buildControls(viewer, demo) {
  const controls = el('div', 'controls');

  for (const control of demo.controls || []) {
    const field = el('label', 'control');
    const readout = el('span', 'control-value', formatValue(control, Number(control.value)));
    const header = el('div', 'control-head');
    header.append(el('span', null, control.label), readout);

    const input = document.createElement('input');
    Object.assign(input, {
      type: 'range', min: control.min, max: control.max, step: control.step, value: control.value
    });
    input.addEventListener('input', () => {
      const value = Number(input.value);
      readout.textContent = formatValue(control, value);
      viewer.setParam(control.id, value);
    });

    viewer.fields.set(control.id, { input, readout, control });
    field.append(header, input);
    controls.append(field);
  }

  return controls;
}

function buildPresets(viewer, presets) {
  const row = el('div', 'presets');

  for (const preset of presets) {
    const button = el('button', 'preset', preset.label);
    button.type = 'button';
    button.addEventListener('click', () => viewer.applyPreset(preset));
    viewer.presetButtons.push({ button, preset });
    row.append(button);
  }

  return row;
}

function buildPlayButton(viewer) {
  const row = el('div', 'presets');
  const button = el('button', 'preset', 'анімація');
  button.type = 'button';
  button.addEventListener('click', () => {
    const playing = viewer.togglePlay();
    button.textContent = playing ? 'зупинити' : 'анімація';
    button.setAttribute('aria-pressed', String(playing));
  });
  row.append(button);
  return row;
}

function buildPanels(viewer, demo) {
  const panels = [];

  if (demo.histogram) {
    const panel = el('div', 'panel');
    const canvas = document.createElement('canvas');
    canvas.className = 'histogram';
    canvas.width = 360;
    canvas.height = 130;
    viewer.histogram = new Histogram(canvas, demo.histogram);

    panel.append(el('div', 'panel-label', 'ГІСТОГРАМА ЗРІЗУ'), canvas);
    if (demo.note) panel.append(el('div', 'panel-note', demo.note));
    panels.push(panel);
  }

  if (demo.readout) {
    const panel = el('div', 'panel');
    viewer.readout = el('div', 'readout');
    panel.append(el('div', 'panel-label', demo.readout.label || 'ВЕЛИЧИНИ'), viewer.readout);
    panels.push(panel);
  }

  return panels;
}

function buildCodeListing(lab) {
  const block = el('div', 'code');
  const header = el('div', 'code-head');
  header.append(el('span', null, lab.code.label || ''), el('span', null, lab.code.library || ''));

  const listing = document.createElement('pre');
  block.append(header, listing);

  fetch(`/api/labs/${encodeURIComponent(lab.id)}/source`)
    .then(response => response.ok ? response.json() : Promise.reject(response.status))
    .then(payload => { listing.innerHTML = highlightPython(payload.text); })
    .catch(() => { listing.textContent = 'лістинг недоступний'; });

  return block;
}

function renderNav() {
  dom.nav.replaceChildren(...app.subjects.map(subject => {
    const open = !app.collapsed.has(subject.id);
    const node = el('div', 'nav-subject');
    node.dataset.open = String(open);
    node.dataset.active = String(subject.items.some(lab => lab.id === app.view));

    const header = el('div', 'nav-subject-head');
    header.append(el('span', 'nav-subject-title', subject.title),
                  el('span', 'nav-subject-sign', subject.sign));
    header.addEventListener('click', () => {
      if (open) app.collapsed.add(subject.id);
      else app.collapsed.delete(subject.id);
      renderNav();
    });

    const items = el('div', 'nav-items');
    for (const lab of subject.items) {
      const item = el('div', 'nav-item');
      item.setAttribute('aria-current', String(lab.id === app.view));
      item.append(el('span', 'nav-item-num', pad(lab.number)),
                  el('span', 'nav-item-title', lab.title));
      item.addEventListener('click', () => go(lab.id));
      items.append(item);
    }

    node.append(header, items);
    return node;
  }));
}

function renderHome() {
  const page = el('div', 'home');
  const header = el('div', 'home-header');
  header.append(
    el('div', 'eyebrow', `${pluralise(app.labs.length, 'робота', 'роботи', 'робіт')} · ` +
                         `${pluralise(app.subjects.length, 'предмет', 'предмети', 'предметів')}`),
    el('h1', 'home-title', 'Звіти з лабораторних')
  );
  page.append(header);

  for (const subject of app.subjects) {
    const section = el('section', 'section');
    const head = el('div', 'section-head');
    head.append(el('h2', 'section-title', subject.title), el('span', 'section-meta', subject.meta));
    section.append(head);

    for (const lab of subject.items) {
      const row = el('div', 'row');
      const body = el('div', 'row-body');
      body.append(el('span', 'row-title', lab.title), el('span', 'row-summary', lab.summary || ''));
      row.append(el('span', 'row-num', pad(lab.number)), body,
                 el('span', 'row-tools', lab.code?.library?.toLowerCase() || '—'));
      row.addEventListener('click', () => go(lab.id));
      section.append(row);
    }

    page.append(section);
  }

  return page;
}

function renderLabPage(lab) {
  const page = el('article', 'lab');
  const head = el('div', 'lab-head');
  const back = el('div', 'back', `← ${subjectOf(lab).title.toUpperCase()} · ${pad(lab.number)}`);
  back.addEventListener('click', () => go(HOME));

  head.append(back, el('h1', 'lab-title', lab.title));
  page.append(head);

  for (const demo of lab.demos) page.append(buildDemo(lab, demo));
  if (lab.code) page.append(buildCodeListing(lab));

  const next = app.labs[app.labs.indexOf(lab) + 1];
  if (next) page.append(buildNextLink(next));

  return page;
}

function buildNextLink(next) {
  const link = el('div', 'next');
  const body = el('div', 'next-body');
  body.append(el('span', 'next-label', `ДАЛІ ${pad(next.number)}`),
              el('span', 'next-title', next.title));
  link.append(body, el('span', 'next-arrow', '→'));
  link.addEventListener('click', () => go(next.id));
  return link;
}

function renderStub(lab) {
  const page = el('article', 'stub');
  const back = el('div', 'back', '← ВСІ ПРЕДМЕТИ');
  back.addEventListener('click', () => go(HOME));

  page.append(back,
              el('div', 'stub-subject', subjectOf(lab).title.toUpperCase()),
              el('h1', 'stub-title', lab.title),
              el('p', 'stub-text', 'Сторінка роботи ще порожня: у ній не оголошено жодної демонстрації.'));

  const slots = el('div', 'stub-slots');
  for (const [label, height] of [['МІСЦЕ ПІД ПЕРЕГЛЯДАЧ', '210px'], ['МІСЦЕ ПІД КОД І ВЕЛИЧИНИ', '120px']]) {
    const slot = el('div', 'stub-slot', label);
    slot.style.height = height;
    slots.append(slot);
  }

  page.append(slots);
  return page;
}

function urlFor(view) {
  if (view === HOME) return '/';
  const lab = app.labs.find(item => item.id === view);
  return lab?.slug ? `/${lab.slug}` : '/';
}

function viewFromUrl() {
  const slug = location.pathname.replace(/^\/+|\/+$/g, '');
  if (!slug) return HOME;
  return app.labs.find(lab => lab.slug === slug)?.id ?? HOME;
}

function show(view) {
  app.view = view;
  for (const viewer of app.viewers) viewer.destroy();
  app.viewers = [];

  renderNav();
  const lab = app.labs.find(item => item.id === view);
  if (!lab) dom.main.replaceChildren(renderHome());
  else dom.main.replaceChildren(lab.demos?.length ? renderLabPage(lab) : renderStub(lab));

  window.scrollTo(0, 0);
}

function go(view) {
  show(view);
  const url = urlFor(view);
  if (url !== location.pathname) history.pushState({ view }, '', url);
}

window.addEventListener('popstate', () => show(viewFromUrl()));
dom.brand.addEventListener('click', () => go(HOME));

fetch('/api/labs')
  .then(response => response.json())
  .then(payload => {
    app.labs = payload.labs;
    app.subjects = groupByCourse(app.labs);

    const view = viewFromUrl();
    if (view === HOME && location.pathname !== '/') history.replaceState(null, '', '/');
    show(view);
  })
  .catch(() => {
    dom.main.replaceChildren(el('div', 'error', 'не вдалося отримати перелік робіт'));
  });
