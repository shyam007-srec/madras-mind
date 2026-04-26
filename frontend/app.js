const corridorEl = document.getElementById('corridor');
const insightGridEl = document.getElementById('insight-grid');
const auditLogEl = document.getElementById('audit-log');
const globalPressureEl = document.getElementById('global-pressure');
const junctionCountEl = document.getElementById('junction-count');
const blindCountEl = document.getElementById('blind-count');
const template = document.getElementById('junction-template');

const junctionOrder = [
  'anna_salai_0',
  'anna_salai_1',
  'anna_salai_2',
  'anna_salai_3',
  'anna_salai_4',
];

const junctionLabels = {
  anna_salai_0: 'Teynampet',
  anna_salai_1: 'Nandanam',
  anna_salai_2: 'Saidapet',
  anna_salai_3: 'Little Mount',
  anna_salai_4: 'Guindy',
};

const phaseNames = ['Phase A', 'Phase B', 'Phase C', 'Phase D'];
const history = [];

function pressureClass(pressure) {
  if (pressure >= 18) {
    return 'critical';
  }
  if (pressure >= 8) {
    return 'warning';
  }
  return 'good';
}

function formatNumber(value, digits = 1) {
  return Number(value || 0).toFixed(digits);
}

function formatLabel(key) {
  return junctionLabels[key] || key;
}

function buildJunctionCard(key, data) {
  const card = template.content.firstElementChild.cloneNode(true);
  const titleEl = card.querySelector('h3');
  const statusEl = card.querySelector('.junction-status');
  const phaseEl = card.querySelector('.phase-pill');
  const pressureValueEl = card.querySelector('.pressure-value');
  const pressureBarEl = card.querySelector('.pressure-bar span');
  const queueValueEl = card.querySelector('.queue-value');
  const occupancyValueEl = card.querySelector('.occupancy-value');
  const neighborPressureValueEl = card.querySelector('.neighbor-pressure-value');

  titleEl.textContent = formatLabel(key);
  statusEl.textContent = data.blind ? 'Sensor blind' : 'Sensor live';
  statusEl.className = `junction-status ${data.blind ? 'blind' : 'good'}`;
  phaseEl.textContent = phaseNames[data.current_phase % phaseNames.length] || `Phase ${data.current_phase}`;
  pressureValueEl.textContent = formatNumber(data.pressure, 1);
  queueValueEl.textContent = formatNumber(data.queue_length, 1);
  occupancyValueEl.textContent = `${formatNumber(data.occupancy_m2, 0)} m²`;
  neighborPressureValueEl.textContent = formatNumber(data.neighbor_inference?.avg_pressure, 1);

  const normalizedPressure = Math.min(100, Math.max(0, data.pressure * 4));
  pressureBarEl.style.width = `${normalizedPressure}%`;
  pressureBarEl.style.filter = `drop-shadow(0 0 10px rgba(86, 209, 255, ${0.25 + normalizedPressure / 220}))`;

  return card;
}

function renderDashboard(snapshot) {
  const junctionEntries = junctionOrder.map((key) => [key, snapshot[key]]).filter(([, value]) => Boolean(value));
  const globalPressure = snapshot.global_pressure?.value ?? 0;
  const blindCount = junctionEntries.filter(([, value]) => value.blind).length;

  globalPressureEl.textContent = formatNumber(globalPressure, 2);
  junctionCountEl.textContent = String(junctionEntries.length);
  blindCountEl.textContent = String(blindCount);

  corridorEl.innerHTML = '';
  junctionEntries.forEach(([key, data], index) => {
    const card = buildJunctionCard(key, data);
    card.style.animationDelay = `${index * 70}ms`;
    corridorEl.appendChild(card);
  });

  insightGridEl.innerHTML = '';
  junctionEntries.slice(0, 4).forEach(([key, data]) => {
    const item = document.createElement('div');
    item.className = 'insight-item';
    item.innerHTML = `
      <div class="insight-dot"></div>
      <div>
        <strong>${formatLabel(key)}</strong>
        <span>${data.blind ? 'Blind via data dropout' : 'Clean sensor stream'}</span>
      </div>
      <b class="${pressureClass(data.pressure)}">${formatNumber(data.pressure, 1)}</b>
    `;
    insightGridEl.appendChild(item);
  });

  const latestEntry = {
    timestamp: new Date(),
    pressure: globalPressure,
    summary: `${junctionEntries.length} intersections live, ${blindCount} blind`,
    details: junctionEntries
      .map(([key, value]) => `${formatLabel(key)}=${formatNumber(value.pressure, 1)}`)
      .join(' | '),
  };
  history.unshift(latestEntry);
  history.splice(6);

  auditLogEl.innerHTML = history
    .map(
      (entry) => `
        <div class="audit-entry">
          <strong>Pressure ${formatNumber(entry.pressure, 2)}</strong>
          <small>${entry.summary}</small>
          <small>${entry.details}</small>
        </div>
      `
    )
    .join('');
}

async function loadInitialSnapshot() {
  const response = await fetch('/metrics', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error('Failed to load metrics');
  }
  renderDashboard(await response.json());
}

function startStream() {
  const source = new EventSource('/stream');
  source.onmessage = (event) => {
    try {
      renderDashboard(JSON.parse(event.data));
    } catch (error) {
      console.error('Invalid stream payload', error);
    }
  };
  source.onerror = async () => {
    source.close();
    try {
      await loadInitialSnapshot();
    } catch (error) {
      console.error(error);
    }
    setTimeout(startStream, 2000);
  };
}

loadInitialSnapshot()
  .catch((error) => {
    console.error(error);
    auditLogEl.innerHTML = `<div class="audit-entry"><strong>Offline</strong><small>Unable to connect to /metrics</small></div>`;
  })
  .finally(() => {
    startStream();
  });
