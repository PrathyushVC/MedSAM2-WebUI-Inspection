'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  fileId: null,
  numSlices: 0,
  currentSlice: 0,
  autoWc: 40,
  autoWw: 400,
  wc: 40,
  ww: 400,
  bbox: null,
  keySlice: null,
  jobId: null,
  resultReady: false,
  showOverlay: true,
  overlayOpacity: 0.7,
  drawing: false,
  dragStart: null,
  imgNaturalW: 512,
  imgNaturalH: 512,
  backend: 'medsam2',         // 'medsam2' | 'totalseg'
  tsTask: 'total_mr',
  tsStructures: new Set(['pancreas']),
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const uploadZone     = document.getElementById('uploadZone');
const fileInput      = document.getElementById('fileInput');
const fileInfoBox    = document.getElementById('fileInfoBox');
const checkpointSel  = document.getElementById('checkpointSelect');
const wcSlider       = document.getElementById('wcSlider');
const wwSlider       = document.getElementById('wwSlider');
const wcVal          = document.getElementById('wcVal');
const wwVal          = document.getElementById('wwVal');
const segBtn         = document.getElementById('segBtn');
const clearBoxBtn    = document.getElementById('clearBoxBtn');
const bboxDisplay    = document.getElementById('bboxDisplay');
const keySliceDisplay= document.getElementById('keySliceDisplay');
const progressArea   = document.getElementById('progressArea');
const progressBar    = document.getElementById('progressBar');
const progressVal    = document.getElementById('progressVal');
const statusMsg      = document.getElementById('statusMsg');
const viewerArea     = document.getElementById('viewerArea');
const placeholder    = document.getElementById('placeholder');
const overlaySlider  = document.getElementById('overlaySlider');
const overlayVal     = document.getElementById('overlayVal');
const resultArea     = document.getElementById('resultArea');
const noResultMsg    = document.getElementById('noResultMsg');
const dlMaskBtn      = document.getElementById('dlMaskBtn');
const dlImageBtn     = document.getElementById('dlImageBtn');
const deviceBadge    = document.getElementById('deviceBadge');
const noCheckpointWarn = document.getElementById('noCheckpointWarn');

// ---------------------------------------------------------------------------
// Canvas elements (created dynamically after upload)
// ---------------------------------------------------------------------------
let canvasContainer, mainCanvas, bboxCanvas, overlayCanvas;
let mainCtx, bboxCtx, overlayCtx;
let sliceBar, sliceInput, sliceNumEl;

function buildViewer(numSlices) {
  viewerArea.innerHTML = '';

  // Slice bar
  sliceBar = document.createElement('div');
  sliceBar.className = 'slice-bar';
  sliceBar.innerHTML = `
    <label>Slice</label>
    <input type="range" id="sliceRange" min="0" max="${numSlices - 1}" value="0">
    <span class="slice-num" id="sliceNum">0</span>
  `;
  viewerArea.appendChild(sliceBar);

  sliceInput = sliceBar.querySelector('#sliceRange');
  sliceNumEl = sliceBar.querySelector('#sliceNum');

  sliceInput.addEventListener('input', () => {
    state.currentSlice = parseInt(sliceInput.value);
    sliceNumEl.textContent = state.currentSlice;
    loadSlice();
    onAxialSliceChange();
  });

  // Canvas container
  canvasContainer = document.createElement('div');
  canvasContainer.className = 'canvas-container';

  mainCanvas    = document.createElement('canvas');
  mainCanvas.id = 'mainCanvas';
  mainCanvas.width = 512; mainCanvas.height = 512;

  bboxCanvas    = document.createElement('canvas');
  bboxCanvas.id = 'bboxCanvas';
  bboxCanvas.width = 512; bboxCanvas.height = 512;

  overlayCanvas    = document.createElement('canvas');
  overlayCanvas.id = 'overlayCanvas';
  overlayCanvas.width = 512; overlayCanvas.height = 512;
  overlayCanvas.style.opacity = state.overlayOpacity;

  canvasContainer.appendChild(mainCanvas);
  canvasContainer.appendChild(overlayCanvas);
  canvasContainer.appendChild(bboxCanvas);
  viewerArea.appendChild(canvasContainer);

  mainCtx    = mainCanvas.getContext('2d');
  bboxCtx    = bboxCanvas.getContext('2d');
  overlayCtx = overlayCanvas.getContext('2d');

  bboxCanvas.addEventListener('mousedown', onMouseDown);
  bboxCanvas.addEventListener('mousemove', onMouseMove);
  bboxCanvas.addEventListener('mouseup',   onMouseUp);

  // Key-board navigation
  document.addEventListener('keydown', e => {
    if (!state.fileId) return;
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') {
      setSlice(state.currentSlice + 1);
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') {
      setSlice(state.currentSlice - 1);
    }
  });
}

function setSlice(idx) {
  idx = Math.max(0, Math.min(idx, state.numSlices - 1));
  state.currentSlice = idx;
  sliceInput.value = idx;
  sliceNumEl.textContent = idx;
  loadSlice();
  onAxialSliceChange();
}

// ---------------------------------------------------------------------------
// Window / Level presets
// ---------------------------------------------------------------------------
const PRESETS = {
  soft:  { wc:  40, ww: 400 },
  lung:  { wc: -600, ww: 1500 },
  bone:  { wc:  400, ww: 1800 },
};

function applyPreset(name) {
  const p = PRESETS[name];
  wcSlider.value = p.wc; wwSlider.value = p.ww;
  state.wc = p.wc; state.ww = p.ww;
  wcVal.textContent = p.wc; wwVal.textContent = p.ww;
  if (state.fileId) loadSlice();
}

wcSlider.addEventListener('input', () => {
  state.wc = parseInt(wcSlider.value);
  wcVal.textContent = state.wc;
  if (state.fileId) { loadSlice(); onWindowLevelChange(); }
});
wwSlider.addEventListener('input', () => {
  state.ww = parseInt(wwSlider.value);
  wwVal.textContent = state.ww;
  if (state.fileId) { loadSlice(); onWindowLevelChange(); }
});

// ---------------------------------------------------------------------------
// Overlay opacity
// ---------------------------------------------------------------------------
overlaySlider.addEventListener('input', () => {
  state.overlayOpacity = parseInt(overlaySlider.value) / 100;
  overlayVal.textContent = overlaySlider.value + '%';
  if (overlayCanvas) overlayCanvas.style.opacity = state.overlayOpacity;
});

function toggleOverlay() {
  state.showOverlay = !state.showOverlay;
  if (overlayCanvas) overlayCanvas.style.display = state.showOverlay ? 'block' : 'none';
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------
const folderInput    = document.getElementById('folderInput');
const browseFilesBtn = document.getElementById('browseFilesBtn');
const browseFolderBtn= document.getElementById('browseFolderBtn');

browseFilesBtn.addEventListener('click',  e => { e.stopPropagation(); fileInput.click(); });
browseFolderBtn.addEventListener('click', e => { e.stopPropagation(); folderInput.click(); });
fileInput.addEventListener('change',   e => handleFiles(e.target.files));
folderInput.addEventListener('change', e => handleFiles(e.target.files));

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', async e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');

  // Try to read folder entries (Chrome/Safari support)
  const items = e.dataTransfer.items;
  if (items && items.length > 0 && items[0].webkitGetAsEntry) {
    const entry = items[0].webkitGetAsEntry();
    if (entry && entry.isDirectory) {
      fileInfoBox.style.display = 'block';
      fileInfoBox.innerHTML = `<div class="fname"><i class="bi bi-hourglass-split me-1"></i>Reading folder…</div>`;
      const allFiles = await readDirectoryEntries(entry);
      if (allFiles.length === 0) {
        fileInfoBox.innerHTML = `<div style="color:#f66">No DICOM files found in folder.</div>`;
        return;
      }
      handleFiles(allFiles);
      return;
    }
  }
  // Fallback: regular file drop
  handleFiles(e.dataTransfer.files);
});

// ---------------------------------------------------------------------------
// Directory reading (for drag-and-drop folder support)
// ---------------------------------------------------------------------------
function readDirectoryEntries(dirEntry) {
  return new Promise(resolve => {
    const files = [];
    const reader = dirEntry.createReader();
    function readBatch() {
      reader.readEntries(async entries => {
        if (!entries.length) { resolve(files); return; }
        for (const entry of entries) {
          if (entry.isFile) {
            const name = entry.name.toLowerCase();
            if (name.endsWith('.dcm') || name.endsWith('.ima') ||
                name.endsWith('.nii') || name.endsWith('.nii.gz') ||
                name.endsWith('.zip') || !name.includes('.')) {
              const file = await new Promise(r => entry.file(r));
              files.push(file);
            }
          } else if (entry.isDirectory) {
            const subFiles = await readDirectoryEntries(entry);
            files.push(...subFiles);
          }
        }
        readBatch(); // read next batch (browsers return max 100 entries at a time)
      });
    }
    readBatch();
  });
}

function detectInputType(files) {
  if (files.length === 0) return null;
  const names = Array.from(files).map(f => f.name.toLowerCase());
  if (names.length === 1 && (names[0].endsWith('.nii.gz') || names[0].endsWith('.nii'))) return 'nifti';
  if (names.length === 1 && names[0].endsWith('.zip')) return 'zip';
  if (names.every(n => n.endsWith('.dcm') || n.endsWith('.ima') || !n.includes('.'))) return 'dicom';
  // mixed — try as DICOM
  return 'dicom';
}

function inputTypeLabel(type) {
  return { nifti: 'NIfTI', zip: 'DICOM ZIP', dicom: 'DICOM series' }[type] || 'file';
}

async function handleFiles(files) {
  if (!files || files.length === 0) return;

  const inputType = detectInputType(files);
  if (!inputType) {
    alert('Unsupported file type. Upload a .nii.gz, .nii, .dcm series, or .zip of DICOMs.');
    return;
  }

  const label = files.length > 1 ? `${files.length} DICOM files` : files[0].name;
  fileInfoBox.style.display = 'block';
  fileInfoBox.innerHTML = `<div class="fname"><i class="bi bi-hourglass-split me-1"></i>
    Uploading ${label}${inputType !== 'nifti' ? ' (converting to NIfTI…)' : '…'}</div>`;

  const formData = new FormData();
  for (const f of files) formData.append('files', f);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      const text = await res.text();
      let msg = `Upload failed (HTTP ${res.status})`;
      try { msg = JSON.parse(text).detail || msg; } catch (_) { if (text.length < 200) msg = text; }
      throw new Error(msg);
    }
    const data = await res.json();

    state.fileId    = data.file_id;
    state.numSlices = data.num_slices;
    state.autoWc    = data.auto_wc;
    state.autoWw    = data.auto_ww;
    state.wc        = data.auto_wc;
    state.ww        = data.auto_ww;
    state.bbox      = null;
    state.keySlice  = null;
    state.resultReady = false;
    mpr.enabled = false;
    mpr.hasResult = false;
    document.getElementById('mprPanel').style.display = 'none';

    wcSlider.value = state.wc; wcVal.textContent = state.wc;
    wwSlider.value = state.ww; wwVal.textContent = state.ww;

    // Build metadata line
    let metaLine = `${data.shape[0]} slices · ${data.shape[1]}×${data.shape[2]} px`;
    if (data.modality)          metaLine += ` · ${data.modality}`;
    if (data.series_description) metaLine += ` · ${data.series_description}`;

    const sourceIcon = { nifti: 'bi-file-earmark-medical', zip: 'bi-file-zip', dicom: 'bi-layers' }[data.source_type] || 'bi-file-medical';
    fileInfoBox.innerHTML = `
      <div class="fname"><i class="bi bi-check-circle-fill me-1" style="color:var(--accent2)"></i>
        <i class="bi ${sourceIcon} me-1" style="color:var(--accent)"></i>${data.filename}</div>
      <div class="fmeta">${metaLine}</div>
    `;

    placeholder.style.display = 'none';
    buildViewer(data.num_slices);
    setSlice(Math.floor(data.num_slices / 2));
    updateSegBtn();
    initMprRaw();

  } catch (err) {
    fileInfoBox.innerHTML = `<div style="color:#f66"><i class="bi bi-x-circle me-1"></i>${err.message}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Load slice image
// ---------------------------------------------------------------------------
async function loadSlice() {
  if (!state.fileId || !mainCtx) return;
  const url = `/api/slice/${state.fileId}/${state.currentSlice}?wc=${state.wc}&ww=${state.ww}`;
  const img = new Image();
  img.onload = () => {
    state.imgNaturalW = img.naturalWidth;
    state.imgNaturalH = img.naturalHeight;
    mainCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
    mainCtx.drawImage(img, 0, 0, mainCanvas.width, mainCanvas.height);
    redrawBbox();
    if (state.resultReady && state.showOverlay) loadResultSlice();
  };
  img.src = url;
}

// ---------------------------------------------------------------------------
// Bbox drawing
// ---------------------------------------------------------------------------
function imgCoords(e) {
  const rect = bboxCanvas.getBoundingClientRect();
  const scaleX = bboxCanvas.width  / rect.width;
  const scaleY = bboxCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left)  * scaleX,
    y: (e.clientY - rect.top)   * scaleY,
  };
}

function onMouseDown(e) {
  if (!state.fileId) return;
  state.drawing = true;
  state.dragStart = imgCoords(e);
  // mark key slice on first draw or redraw
  state.keySlice = state.currentSlice;
  keySliceDisplay.textContent = `key slice: ${state.keySlice}`;
}

function onMouseMove(e) {
  if (!state.drawing) return;
  const cur = imgCoords(e);
  const sx = state.dragStart.x, sy = state.dragStart.y;
  bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
  drawBboxRect(sx, sy, cur.x - sx, cur.y - sy);
}

function onMouseUp(e) {
  if (!state.drawing) return;
  state.drawing = false;
  const cur = imgCoords(e);
  const sx = state.dragStart.x, sy = state.dragStart.y;

  // Normalise to (x0 < x1)
  const x0 = Math.min(sx, cur.x), y0 = Math.min(sy, cur.y);
  const x1 = Math.max(sx, cur.x), y1 = Math.max(sy, cur.y);

  if (Math.abs(x1 - x0) < 4 || Math.abs(y1 - y0) < 4) {
    state.drawing = false; return;
  }

  // Store in original image coords (backend expects these)
  const scaleX = state.imgNaturalW / bboxCanvas.width;
  const scaleY = state.imgNaturalH / bboxCanvas.height;
  state.bbox = {
    x0: Math.round(x0 * scaleX), y0: Math.round(y0 * scaleY),
    x1: Math.round(x1 * scaleX), y1: Math.round(y1 * scaleY),
  };

  bboxDisplay.textContent = `[${state.bbox.x0}, ${state.bbox.y0}, ${state.bbox.x1}, ${state.bbox.y1}]`;
  clearBoxBtn.style.display = 'block';
  redrawBbox();
  updateSegBtn();
}

function redrawBbox() {
  if (!bboxCtx) return;
  bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
  if (!state.bbox) return;
  const scaleX = bboxCanvas.width  / state.imgNaturalW;
  const scaleY = bboxCanvas.height / state.imgNaturalH;
  const cx0 = state.bbox.x0 * scaleX, cy0 = state.bbox.y0 * scaleY;
  const cw  = (state.bbox.x1 - state.bbox.x0) * scaleX;
  const ch  = (state.bbox.y1 - state.bbox.y0) * scaleY;
  drawBboxRect(cx0, cy0, cw, ch);
}

function drawBboxRect(x, y, w, h) {
  bboxCtx.strokeStyle = '#00bfff';
  bboxCtx.lineWidth = 2;
  bboxCtx.setLineDash([6, 3]);
  bboxCtx.strokeRect(x, y, w, h);
  bboxCtx.fillStyle = 'rgba(0,191,255,0.06)';
  bboxCtx.fillRect(x, y, w, h);
}

clearBoxBtn.addEventListener('click', () => {
  state.bbox = null;
  state.keySlice = null;
  bboxDisplay.textContent = '—';
  keySliceDisplay.textContent = 'slice —';
  if (bboxCtx) bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
  clearBoxBtn.style.display = 'none';
  updateSegBtn();
});

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------
function updateSegBtn() {
  const hasFile = !!state.fileId;
  if (state.backend === 'medsam2') {
    const hasBbox = !!state.bbox;
    const hasCkpt = checkpointSel.value !== '';
    segBtn.disabled = !(hasFile && hasBbox && hasCkpt);
    segBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i>Segment';
  } else {
    const hasStructs = state.tsStructures.size > 0;
    segBtn.disabled = !(hasFile && hasStructs);
    segBtn.innerHTML = '<i class="bi bi-magic me-1"></i>Auto-Segment';
  }
}

checkpointSel.addEventListener('change', updateSegBtn);

segBtn.addEventListener('click', startSegmentation);

async function startSegmentation() {
  if (!state.fileId) return;
  segBtn.disabled = true;
  progressArea.style.display = 'block';
  setProgress(0, 'Queuing…');

  try {
    let res;
    if (state.backend === 'medsam2') {
      if (!state.bbox || !checkpointSel.value) return;
      res = await fetch('/api/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_id: state.fileId,
          key_slice: state.keySlice ?? state.currentSlice,
          bbox: [state.bbox.x0, state.bbox.y0, state.bbox.x1, state.bbox.y1],
          checkpoint: checkpointSel.value,
          wc: state.wc,
          ww: state.ww,
        }),
      });
    } else {
      res = await fetch('/api/segment/totalseg', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_id: state.fileId,
          task: state.tsTask,
          structures: [...state.tsStructures],
          wc: state.wc,
          ww: state.ww,
        }),
      });
    }

    if (!res.ok) {
      const text = await res.text();
      let msg = `Segment failed (HTTP ${res.status})`;
      try { msg = JSON.parse(text).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const data = await res.json();
    state.jobId = data.job_id;
    pollStatus();
  } catch (err) {
    setProgress(0, `Error: ${err.message}`, true);
    segBtn.disabled = false;
  }
}

let pollTimer = null;
function pollStatus() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (!state.jobId) return;
    try {
      const res = await fetch(`/api/status/${state.jobId}`);
      const data = await res.json();

      setProgress(data.progress, statusLabel(data.status, data.progress));

      if (data.status === 'done') {
        state.resultReady = true;
        setProgress(100, 'Segmentation complete');
        showResults(data);
        overlaySlider.disabled = false;
        loadResultSlice();
        segBtn.disabled = false;
        initMprResult();
      } else if (data.status === 'error') {
        setProgress(0, `Error: ${data.error}`, true);
        segBtn.disabled = false;
      } else {
        pollStatus();
      }
    } catch (err) {
      pollStatus();
    }
  }, 1200);
}

function statusLabel(status, pct) {
  if (state.backend === 'totalseg') {
    const labels = {
      queued:  'Waiting in queue…',
      running: pct < 20  ? 'Loading file…'
             : pct < 80  ? 'Running TotalSegmentator (this takes 1–5 min on CPU)…'
             : 'Saving results…',
      done:    'Done',
      error:   'Error',
    };
    return labels[status] || status;
  }
  const labels = {
    queued:  'Waiting in queue…',
    running: pct < 40  ? 'Loading file…'
           : pct < 55  ? 'Preparing tensors…'
           : pct < 80  ? 'Running MedSAM2 inference…'
           : 'Saving results…',
    done:    'Done',
    error:   'Error',
  };
  return labels[status] || status;
}

function setProgress(pct, label, isError = false) {
  progressBar.style.width = pct + '%';
  progressBar.style.background = isError ? '#f66' : 'var(--accent)';
  progressVal.textContent = pct + '%';
  statusMsg.textContent = label;
}

function showResults(data) {
  noResultMsg.style.display = 'none';
  resultArea.style.display = 'block';
  document.getElementById('statVoxels').textContent = data.voxels?.toLocaleString() ?? '—';
  document.getElementById('statSlices').textContent = data.num_slices ?? '—';
  dlMaskBtn.href  = `/api/download/${state.jobId}/mask`;
  dlImageBtn.href = `/api/download/${state.jobId}/image`;
  dlMaskBtn.setAttribute('download',  'segmentation.nii.gz');
  dlImageBtn.setAttribute('download', 'image_windowed.nii.gz');

  // Legend for TotalSegmentator multi-label
  const legendArea = document.getElementById('legendArea');
  const legendItems = document.getElementById('legendItems');
  if (data.mask_type === 'multilabel' && data.structures?.length) {
    legendArea.style.display = 'block';
    legendItems.innerHTML = '';
    for (const s of data.structures) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:6px';
      const dot = document.createElement('span');
      dot.style.cssText = `width:10px;height:10px;border-radius:50%;background:${structColor(s)};flex-shrink:0`;
      const lbl = document.createElement('span');
      lbl.textContent = s.replace(/_/g, ' ');
      lbl.style.color = 'var(--text)';
      row.appendChild(dot); row.appendChild(lbl);
      legendItems.appendChild(row);
    }
  } else {
    legendArea.style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// Result overlay
// ---------------------------------------------------------------------------
async function loadResultSlice() {
  if (!state.jobId || !state.resultReady || !overlayCtx) return;
  const url = `/api/result/slice/${state.jobId}/${state.currentSlice}`;
  const img = new Image();
  img.onload = () => {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.drawImage(img, 0, 0, overlayCanvas.width, overlayCanvas.height);
  };
  img.src = url;
}

// ---------------------------------------------------------------------------
// MPR (Multi-Planar Reconstruction) views
// ---------------------------------------------------------------------------
const mpr = {
  enabled: false,
  hasResult: false,          // true once a segmentation is done
  dims: { D: 1, H: 1, W: 1 },
  coronal:  { idx: 0 },
  sagittal: { idx: 0 },
};

let mprDebounceTimer = null;

// Called after a file is uploaded — shows raw image MPR immediately
function initMprRaw() {
  if (!state.fileId) return;
  mpr.hasResult = false;
  mpr.dims = { D: state.numSlices, H: 128, W: 128 };  // placeholder; updated by first image load
  mpr.coronal.idx  = Math.floor(state.numSlices / 2);
  mpr.sagittal.idx = 64;
  mpr.enabled = true;
  document.getElementById('mprPanel').style.display = 'block';
  updateMprPositionLabels();
  refreshMpr('coronal');
  refreshMpr('sagittal');
}

// Called after segmentation completes — switches to result overlay
async function initMprResult() {
  if (!state.jobId) return;
  try {
    const res = await fetch(`/api/result/dims/${state.jobId}`);
    if (!res.ok) return;
    const dims = await res.json();
    mpr.dims = dims;
    mpr.coronal.idx  = Math.floor(dims.H / 2);
    mpr.sagittal.idx = Math.floor(dims.W / 2);
    mpr.hasResult = true;
    mpr.enabled = true;
    document.getElementById('mprPanel').style.display = 'block';
    updateMprPositionLabels();
    refreshMpr('coronal');
    refreshMpr('sagittal');
  } catch (e) { console.error('MPR result init failed', e); }
}

function updateMprPositionLabels() {
  const { H, W } = mpr.dims;
  document.getElementById('mprCoronalPos').textContent
    = `y ${mpr.coronal.idx + 1}/${H}`;
  document.getElementById('mprSagittalPos').textContent
    = `x ${mpr.sagittal.idx + 1}/${W}`;
}

function stepMpr(plane, delta) {
  if (!mpr.enabled) return;
  const max = plane === 'coronal' ? mpr.dims.H - 1 : mpr.dims.W - 1;
  mpr[plane].idx = Math.max(0, Math.min(mpr[plane].idx + delta, max));
  updateMprPositionLabels();
  refreshMpr(plane);
  // Updating one plane redraws the other's crosshair
  refreshMpr(plane === 'coronal' ? 'sagittal' : 'coronal', true);
}

function refreshMpr(plane, debounce = false) {
  if (!mpr.enabled) return;
  if (debounce) {
    clearTimeout(mprDebounceTimer);
    mprDebounceTimer = setTimeout(() => { _doRefreshMpr('coronal'); _doRefreshMpr('sagittal'); }, 120);
  } else {
    _doRefreshMpr(plane);
  }
}

function _doRefreshMpr(plane) {
  if (!mpr.enabled) return;
  const img     = document.getElementById(plane === 'coronal' ? 'mprCoronalImg'     : 'mprSagittalImg');
  const spinner = document.getElementById(plane === 'coronal' ? 'mprCoronalSpinner' : 'mprSagittalSpinner');
  const sliceIdx = plane === 'coronal' ? mpr.coronal.idx : mpr.sagittal.idx;

  let url;
  if (mpr.hasResult && state.jobId) {
    url = `/api/result/mpr/${state.jobId}/${plane}/${sliceIdx}`
        + `?axial_pos=${state.currentSlice}`;
    if (plane === 'coronal')  url += `&sagittal_pos=${mpr.sagittal.idx}`;
    if (plane === 'sagittal') url += `&coronal_pos=${mpr.coronal.idx}`;
  } else if (state.fileId) {
    url = `/api/mpr/${state.fileId}/${plane}/${sliceIdx}`
        + `?wc=${state.wc}&ww=${state.ww}&axial_pos=${state.currentSlice}`;
    if (plane === 'coronal')  url += `&sagittal_pos=${mpr.sagittal.idx}`;
    if (plane === 'sagittal') url += `&coronal_pos=${mpr.coronal.idx}`;
  } else {
    return;
  }

  spinner.classList.add('active');
  const tmp = new Image();
  tmp.onload = () => {
    // Update dims from image natural size for the first load
    img.src = tmp.src;
    spinner.classList.remove('active');
  };
  tmp.onerror = () => spinner.classList.remove('active');
  tmp.src = url + `&_t=${Date.now()}`;
}

// Called whenever the axial slice changes — updates crosshairs in both planes
function onAxialSliceChange() {
  if (!mpr.enabled) return;
  clearTimeout(mprDebounceTimer);
  mprDebounceTimer = setTimeout(() => {
    _doRefreshMpr('coronal');
    _doRefreshMpr('sagittal');
  }, 100);
}

// Also refresh raw MPR when window/level changes
function onWindowLevelChange() {
  if (mpr.enabled && !mpr.hasResult) {
    refreshMpr('coronal',  true);
    refreshMpr('sagittal', true);
  }
}

// Patch setSlice to also load overlay
const origLoadSlice = loadSlice;

// ---------------------------------------------------------------------------
// Backend toggle
// ---------------------------------------------------------------------------
const panelMedSAM2  = document.getElementById('panelMedSAM2');
const panelTotalSeg = document.getElementById('panelTotalSeg');
const btnMedSAM2    = document.getElementById('btnBackendMedSAM2');
const btnTotalSeg   = document.getElementById('btnBackendTotalSeg');
const tsTaskSelect  = document.getElementById('tsTaskSelect');
const tsStructCount = document.getElementById('tsStructCount');
const tsStructGroups= document.getElementById('tsStructGroups');

function setBackend(backend) {
  state.backend = backend;
  if (backend === 'medsam2') {
    panelMedSAM2.style.display  = '';
    panelTotalSeg.style.display = 'none';
    clearBoxBtn.style.display = state.bbox ? 'block' : 'none';
    btnMedSAM2.classList.add('active');
    btnTotalSeg.classList.remove('active');
  } else {
    panelMedSAM2.style.display  = 'none';
    panelTotalSeg.style.display = '';
    clearBoxBtn.style.display   = 'none';
    btnMedSAM2.classList.remove('active');
    btnTotalSeg.classList.add('active');
  }
  updateSegBtn();
}

tsTaskSelect.addEventListener('change', () => {
  state.tsTask = tsTaskSelect.value;
  state.tsStructures = new Set(['pancreas']);
  loadTsStructures();
  updateSegBtn();
});

// ---------------------------------------------------------------------------
// TotalSegmentator structure picker
// ---------------------------------------------------------------------------

// Map structure name → RGB colour (mirrors Python backend)
const STRUCT_COLORS = {
  pancreas:               [255,200,0],
  liver:                  [100,200,100],
  spleen:                 [100,150,255],
  kidney_right:           [255,100,100],
  kidney_left:            [255,130,130],
  gallbladder:            [180,255,100],
  stomach:                [255,160,80],
  duodenum:               [200,100,255],
  small_bowel:            [255,220,150],
  colon:                  [200,160,100],
  adrenal_gland_right:    [255,80,200],
  adrenal_gland_left:     [255,120,220],
  urinary_bladder:        [80,220,255],
  prostate:               [255,80,80],
  heart:                  [255,60,60],
  aorta:                  [255,100,100],
  lung_left:              [150,200,255],
  lung_right:             [150,200,255],
  esophagus:              [220,180,255],
  spinal_cord:            [255,255,100],
};
function structColor(name) {
  const c = STRUCT_COLORS[name] || [200,200,200];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

async function loadTsStructures() {
  try {
    const res = await fetch(`/api/totalseg/structures/${state.tsTask}`);
    if (!res.ok) return;
    const data = await res.json();
    renderStructGroups(data.groups);
  } catch (e) { console.error(e); }
}

function renderStructGroups(groups) {
  tsStructGroups.innerHTML = '';
  for (const [group, structs] of Object.entries(groups)) {
    if (!structs.length) continue;
    const header = document.createElement('div');
    header.style.cssText = 'color:var(--muted);font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin:6px 0 3px';
    header.textContent = group;
    tsStructGroups.appendChild(header);

    for (const s of structs) {
      const row = document.createElement('label');
      row.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;padding:2px 0';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = s;
      cb.checked = state.tsStructures.has(s);
      cb.style.accentColor = structColor(s);
      cb.addEventListener('change', () => {
        if (cb.checked) state.tsStructures.add(s);
        else state.tsStructures.delete(s);
        tsStructCount.textContent = `${state.tsStructures.size} selected`;
        updateSegBtn();
      });
      const dot = document.createElement('span');
      dot.style.cssText = `width:9px;height:9px;border-radius:50%;background:${structColor(s)};flex-shrink:0`;
      const label = document.createElement('span');
      label.textContent = s.replace(/_/g, ' ');
      label.style.color = 'var(--text)';
      row.appendChild(cb); row.appendChild(dot); row.appendChild(label);
      tsStructGroups.appendChild(row);
    }
  }
  tsStructCount.textContent = `${state.tsStructures.size} selected`;
}

// ---------------------------------------------------------------------------
// Checkpoint list
// ---------------------------------------------------------------------------
async function loadCheckpoints() {
  try {
    const res = await fetch('/api/checkpoints');
    const data = await res.json();
    checkpointSel.innerHTML = '<option value="">— select checkpoint —</option>';
    if (data.checkpoints.length === 0) {
      noCheckpointWarn.style.display = 'block';
    } else {
      noCheckpointWarn.style.display = 'none';
      data.checkpoints.forEach(ck => {
        const opt = document.createElement('option');
        opt.value = ck;
        opt.textContent = ck;
        if (ck === 'MedSAM2_latest.pt') opt.selected = true;
        checkpointSel.appendChild(opt);
      });
    }
  } catch (e) {
    noCheckpointWarn.style.display = 'block';
  }
}

// ---------------------------------------------------------------------------
// Device detection (just read from a lightweight status check)
// ---------------------------------------------------------------------------
async function detectDevice() {
  try {
    // We can't directly get device without a server call; show a placeholder
    deviceBadge.textContent = 'CUDA/MPS/CPU auto';
    deviceBadge.className = 'badge bg-info text-dark';
  } catch (e) {}
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
(async function init() {
  await loadCheckpoints();
  await detectDevice();
  await loadTsStructures();
  wcVal.textContent = wcSlider.value;
  wwVal.textContent = wwSlider.value;
})();
