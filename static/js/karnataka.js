// Karnataka Road Accident Severity Frontend Logic

let kMap = null;
let kGeoJsonLayer = null;
let kMarkerLayer = null;

document.addEventListener('DOMContentLoaded', () => {
    initKarnatakaMap();
    initKarnatakaCharts();
    initKarnatakaFilters();
    initKarnatakaForm();
});

function initKarnatakaMap() {
    const mapEl = document.getElementById('karnatakaMap');
    if (!mapEl) return;

    // Centered on Karnataka
    kMap = L.map('karnatakaMap').setView([15.3173, 75.7139], 7);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(kMap);

    kMarkerLayer = L.layerGroup().addTo(kMap);

    // 1. Fetch and overlay Karnataka District Boundaries
    fetch('/api/karnataka/geojson')
        .then(resp => resp.json())
        .then(geoData => {
            if (geoData && geoData.features && geoData.features.length > 0) {
                kGeoJsonLayer = L.geoJSON(geoData, {
                    style: {
                        color: '#2563eb',
                        weight: 1.5,
                        opacity: 0.6,
                        fillColor: '#93c5fd',
                        fillOpacity: 0.1
                    },
                    onEachFeature: (feature, layer) => {
                        const distName = feature.properties?.DISTRICT || feature.properties?.name || 'District';
                        layer.bindTooltip(distName, { sticky: true, className: 'small' });
                    }
                }).addTo(kMap);
            }
        })
        .catch(err => console.error('Failed to load GeoJSON:', err));

    // 2. Render initial incident markers
    const initialMarkers = window.karnatakaData?.markers || [];
    renderKarnatakaMarkers(initialMarkers);
}

function renderKarnatakaMarkers(markers) {
    if (!kMarkerLayer) return;
    kMarkerLayer.clearLayers();

    markers.forEach(m => {
        if (!m.lat || !m.lon) return;

        let color = '#10b981'; // Simple Injury
        if (m.severity === 'Fatal') {
            color = '#ef4444';
        } else if (m.severity === 'Grievous Injury') {
            color = '#f59e0b';
        }

        const circle = L.circleMarker([m.lat, m.lon], {
            radius: m.severity === 'Fatal' ? 4 : 3,
            fillColor: color,
            color: '#ffffff',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.75
        });

        const popup = `
            <div style="font-size: 0.85rem; min-width: 170px;">
                <div class="fw-bold text-slate-900 mb-1">${m.district} (${m.year})</div>
                <div class="mb-1">
                    <span class="badge ${m.severity === 'Fatal' ? 'bg-danger' : (m.severity === 'Grievous Injury' ? 'bg-warning text-dark' : 'bg-success')}">
                        ${m.severity}
                    </span>
                </div>
                <hr class="my-1">
                <div><strong>Collision:</strong> ${m.collision_type}</div>
                <div><strong>Weather:</strong> ${m.weather}</div>
                <div><strong>Vehicles:</strong> ${m.vehicles}</div>
            </div>
        `;
        circle.bindPopup(popup);
        kMarkerLayer.addLayer(circle);
    });

    const countEl = document.getElementById('karnatakaMarkerCount');
    if (countEl) {
        countEl.textContent = `Displaying ${markers.length} mapped incidents`;
    }
}

function initKarnatakaCharts() {
    const data = window.karnatakaData || {};

    // 1. Collision Types Chart
    const colCtx = document.getElementById('collisionChart');
    if (colCtx && data.collisionTypes) {
        new Chart(colCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(data.collisionTypes),
                datasets: [{
                    label: 'Incidents Recorded',
                    data: Object.values(data.collisionTypes),
                    backgroundColor: ['#ef4444', '#f59e0b', '#2563eb', '#10b981'],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }

    // 2. Weather Distribution Chart
    const wCtx = document.getElementById('karnatakaWeatherChart');
    if (wCtx && data.weatherCounts) {
        new Chart(wCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(data.weatherCounts),
                datasets: [{
                    data: Object.values(data.weatherCounts),
                    backgroundColor: ['#38bdf8', '#64748b', '#0284c7', '#f59e0b'],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }
}

function initKarnatakaFilters() {
    const filterBtn = document.getElementById('karnatakaFilterBtn');
    const resetBtn = document.getElementById('karnatakaResetBtn');

    if (!filterBtn) return;

    filterBtn.addEventListener('click', async () => {
        const district = document.getElementById('filterDistrict').value;
        const severity = document.getElementById('filterSeverity').value;
        const year = document.getElementById('filterYear').value;

        filterBtn.disabled = true;
        filterBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Updating...';

        try {
            const url = `/api/karnataka/filter?district=${encodeURIComponent(district)}&severity=${encodeURIComponent(severity)}&year=${encodeURIComponent(year)}`;
            const resp = await fetch(url);
            const resData = await resp.json();

            if (resData.error) {
                showToast(resData.error, 'danger');
                return;
            }

            renderKarnatakaMarkers(resData.markers);
            showToast(`Loaded ${resData.count} incident markers`, 'success');
        } catch (err) {
            console.error(err);
            showToast('Failed to filter map incidents.', 'danger');
        } finally {
            filterBtn.disabled = false;
            filterBtn.innerHTML = '<i class="bi bi-check-lg me-1"></i> Update Map View';
        }
    });

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            document.getElementById('filterDistrict').value = 'All';
            document.getElementById('filterSeverity').value = 'All';
            document.getElementById('filterYear').value = 'All';
            filterBtn.click();
        });
    }
}

function initKarnatakaForm() {
    const form = document.getElementById('karnatakaPredictForm');
    const btn = document.getElementById('karnatakaPredictBtn');
    const placeholder = document.getElementById('kResultsPlaceholder');
    const content = document.getElementById('kResultsContent');

    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span> Classifying...';

        const payload = {
            district: document.getElementById('kp_district').value,
            year: parseInt(document.getElementById('kp_year').value),
            collision_type: document.getElementById('kp_collision').value,
            weather: document.getElementById('kp_weather').value,
            vehicles: parseInt(document.getElementById('kp_vehicles').value),
            hit_run: parseInt(document.getElementById('kp_hit_run').value),
            main_cause: document.getElementById('kp_cause').value,
            latitude: parseFloat(document.getElementById('kp_lat').value),
            longitude: parseFloat(document.getElementById('kp_lon').value)
        };

        try {
            const resp = await fetch('/api/predict/karnataka', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await resp.json();

            if (data.error) {
                showToast(data.error, 'danger');
                return;
            }

            placeholder.style.display = 'none';
            content.style.display = 'block';

            const badge = document.getElementById('kResBadge');
            badge.textContent = data.prediction;
            badge.className = `badge px-3 py-2 fs-6 ${data.badge_class}`;

            // Probability bars
            const fatalVal = data.probabilities['Fatal'] || 0;
            const grievousVal = data.probabilities['Grievous Injury'] || 0;
            const simpleVal = data.probabilities['Simple Injury'] || 0;

            document.getElementById('kProbFatal').textContent = `${fatalVal}%`;
            document.getElementById('kBarFatal').style.width = `${fatalVal}%`;

            document.getElementById('kProbGrievous').textContent = `${grievousVal}%`;
            document.getElementById('kBarGrievous').style.width = `${grievousVal}%`;

            document.getElementById('kProbSimple').textContent = `${simpleVal}%`;
            document.getElementById('kBarSimple').style.width = `${simpleVal}%`;

            // Risk Insights
            const riskUl = document.getElementById('kRiskList');
            riskUl.innerHTML = '';
            (data.risk_insights || []).forEach(item => {
                const li = document.createElement('li');
                li.textContent = item;
                riskUl.appendChild(li);
            });

            showToast(`Karnataka Severity: ${data.prediction}`, 'success');

        } catch (err) {
            console.error(err);
            showToast('Failed to calculate Karnataka severity prediction.', 'danger');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-lightning-charge me-1"></i> Predict Karnataka Severity';
        }
    });
}
