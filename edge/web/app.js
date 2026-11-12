const $ = id => document.getElementById(id);
const canvas = $('twin');
const ctx = canvas.getContext('2d');
const uploadForm = $('upload-form');
const uploadInput = $('upload-video');
const uploadButton = $('upload-submit');
const uploadStatus = $('upload-status');
const imageForm = $('image-form');
const imageInput = $('upload-image');
const imageButton = $('image-submit');
const imageStatus = $('image-status');
let items = new Map();
let seenDecisions = new Set();
let lanes = ['review'];
let currentRunId = null;
let uploading = false;
let imageUploading = false;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

function update(status) {
  if (currentRunId !== status.run_id) {
    if (currentRunId !== null) {
      $('feed-image').src = '/stream.mjpg?run=' + encodeURIComponent(status.run_id);
    }
    currentRunId = status.run_id;
    items.clear();
    seenDecisions.clear();
    lanes = ['review'];
  }

  $('state').textContent = status.error ? 'Error' :
    status.idle ? 'Ready' : status.finished ? 'Finished' : 'Running';
  $('state').title = status.error || '';
  $('feedmode').textContent = status.idle ? '● WAITING' :
    status.finished ? '● COMPLETE' : '● PROCESSING';
  $('fps').textContent = status.idle ? '—' : status.fps.toFixed(1);
  $('p50').textContent = status.pipeline_p50_ms ?? '—';
  $('p95').textContent = status.pipeline_p95_ms ?? '—';
  $('count').textContent = status.material_policy_enabled ?
    status.decisions_total + ' proposed outcomes · ' + status.pending_tracks + ' pending' :
    status.duplicate_gate_enabled ?
      status.decisions_total + ' proposed · ' + status.suppressed_duplicate_candidates + ' duplicate candidates held' :
      status.decisions_total + ' proposed decisions';
  const busy = !status.idle && !status.finished;
  uploadButton.disabled = uploading || imageUploading || busy;
  imageButton.disabled = uploading || imageUploading || busy;

  if (!uploading && status.error) {
    uploadStatus.textContent = 'Processing stopped: ' + status.error;
  } else if (!uploading && status.idle) {
    uploadStatus.textContent = 'Choose an MP4 to run inference on the Jetson.';
  } else if (!uploading && !status.finished) {
    uploadStatus.textContent = 'The Jetson is processing frames. You can upload another video when it finishes.';
  }

  const now = performance.now();
  status.recent_events.forEach((decision, index) => {
    const key = decision.track_id === null ?
      'untracked-' + decision.decision_frame + '-' + decision.class + '-' +
        (decision.bbox_xyxy || []).join(',') :
      decision.track_id;
    if (seenDecisions.has(key)) return;
    seenDecisions.add(key);
    const stagger = status.finished ?
      (status.recent_events.length - 1 - index) * 1400 : 0;
    items.set(key, {label: decision.class, lane: decision.lane, born: now - stagger});
  });
  for (const [key, item] of items) {
    if (now - item.born > 7000) items.delete(key);
  }
  lanes = [...new Set(['review', ...status.recent_events.map(e => e.lane)])].slice(0, 5);

  $('rows').innerHTML = status.recent_events.slice(-20).reverse().map(e =>
    '<div class="row"><span>#' + escapeHtml(e.track_id ?? '?') +
    '</span><span>' + escapeHtml(e.class) +
    (e.reason ? ' · ' + escapeHtml(e.reason.replaceAll('_', ' ')) : '') +
    '</span><span>' + Math.round(e.confidence * 100) +
    '%</span><b>' + escapeHtml(e.lane) + '</b></div>'
  ).join('') || '<p>Waiting for detections…</p>';
}

uploadForm.addEventListener('submit', event => {
  event.preventDefault();
  const file = uploadInput.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.mp4')) {
    uploadStatus.textContent = 'Choose an MP4 video.';
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    uploadStatus.textContent = 'The video exceeds the 200 MiB limit.';
    return;
  }

  uploading = true;
  uploadButton.disabled = true;
  uploadStatus.textContent = 'Uploading ' + file.name + '…';
  const request = new XMLHttpRequest();
  request.open('POST', '/api/upload-video');
  request.setRequestHeader('Content-Type', 'video/mp4');
  request.upload.onprogress = event => {
    if (event.lengthComputable) {
      uploadStatus.textContent = 'Uploading ' + file.name + ': ' +
        Math.round(100 * event.loaded / event.total) + '%';
    }
  };
  request.onload = () => {
    uploading = false;
    if (request.status === 202) {
      uploadStatus.textContent = 'Upload complete. The Jetson is processing ' + file.name + '.';
      uploadInput.value = '';
    } else {
      let message = 'Upload failed.';
      try { message = JSON.parse(request.responseText).detail || message; } catch (_) {}
      uploadStatus.textContent = message;
      uploadButton.disabled = false;
    }
  };
  request.onerror = () => {
    uploading = false;
    uploadStatus.textContent = 'Upload failed. Check the dashboard connection.';
    uploadButton.disabled = false;
  };
  request.send(file);
});

imageForm.addEventListener('submit', event => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file) return;
  const extension = file.name.toLowerCase().split('.').pop();
  const mediaType = extension === 'png' ? 'image/png' :
    ['jpg', 'jpeg'].includes(extension) ? 'image/jpeg' : null;
  if (!mediaType) {
    imageStatus.textContent = 'Choose a JPEG or PNG image.';
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    imageStatus.textContent = 'The image exceeds the 10 MiB limit.';
    return;
  }

  imageUploading = true;
  imageButton.disabled = true;
  uploadButton.disabled = true;
  $('image-result').hidden = true;
  imageStatus.textContent = 'Uploading ' + file.name + '…';
  const request = new XMLHttpRequest();
  request.open('POST', '/api/upload-image');
  request.setRequestHeader('Content-Type', mediaType);
  request.upload.onprogress = event => {
    if (event.lengthComputable) {
      imageStatus.textContent = 'Uploading ' + file.name + ': ' +
        Math.round(100 * event.loaded / event.total) + '%';
    }
  };
  request.onload = () => {
    imageUploading = false;
    if (request.status === 200) {
      const result = JSON.parse(request.responseText);
      $('image-preview').src = 'data:image/jpeg;base64,' + result.annotated_jpeg_base64;
      $('image-summary').textContent = result.detections.length +
        ' detections · ' + result.inference_ms + ' ms prediction';
      $('image-detections').innerHTML = result.detections.map(d =>
        '<div class="image-detection"><span>' + escapeHtml(d.class) +
        ' · score ' + Number(d.confidence).toFixed(2) +
        '</span><b>' + escapeHtml(d.lane) + '</b></div>'
      ).join('') || '<p>No items detected above the configured score threshold.</p>';
      $('image-result').hidden = false;
      imageStatus.textContent = 'Analysis complete. Your photo was not saved.';
      imageInput.value = '';
    } else {
      let message = 'Image analysis failed.';
      try { message = JSON.parse(request.responseText).detail || message; } catch (_) {}
      imageStatus.textContent = message;
    }
    imageButton.disabled = false;
    uploadButton.disabled = false;
  };
  request.onerror = () => {
    imageUploading = false;
    imageStatus.textContent = 'Image upload failed. Check the dashboard connection.';
    imageButton.disabled = false;
    uploadButton.disabled = false;
  };
  request.send(file);
});

const feed = new EventSource('/api/events');
feed.onmessage = event => update(JSON.parse(event.data));
feed.onerror = () => { $('state').textContent = 'Reconnecting'; };

function draw() {
  const width = canvas.width;
  const height = canvas.height;
  const now = performance.now();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#10251d';
  ctx.fillRect(40, 160, width - 80, 150);
  ctx.strokeStyle = '#527d59';
  ctx.lineWidth = 3;
  ctx.strokeRect(40, 160, width - 80, 150);
  for (let x = 65; x < width - 40; x += 58) {
    ctx.strokeStyle = '#315342';
    ctx.beginPath();
    ctx.moveTo(x, 164);
    ctx.lineTo(x, 306);
    ctx.stroke();
  }
  const count = lanes.length;
  lanes.forEach((lane, index) => {
    const x = 80 + index * (width - 160) / count;
    ctx.fillStyle = '#1c392c';
    ctx.fillRect(x, 355, (width - 180) / count, 95);
    ctx.strokeStyle = '#a8f15d';
    ctx.strokeRect(x, 355, (width - 180) / count, 95);
    ctx.fillStyle = '#cbe1ce';
    ctx.font = '15px sans-serif';
    ctx.fillText(lane.toUpperCase(), x + 12, 410);
  });
  for (const item of items.values()) {
    const age = (now - item.born) / 7000;
    if (age > 1) continue;
    const laneIndex = Math.max(0, lanes.indexOf(item.lane));
    const target = 80 + laneIndex * (width - 160) / count + (width - 180) / count / 2;
    const x = 60 + Math.min(age * 1.7, 1) * (target - 60);
    const y = 235 + (age > .6 ? (age - .6) * 390 : 0);
    ctx.fillStyle = '#a8f15d';
    ctx.beginPath();
    ctx.arc(x, y, 17, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e9f1ef';
    ctx.font = '12px sans-serif';
    ctx.fillText(item.label.slice(0, 18), x + 23, y + 4);
  }
  requestAnimationFrame(draw);
}
draw();
