// Bengaluru Accident Hotspot Frontend Logic

let bMap = null;
let wardLayerGroup = null;
let alertLayerGroup = null;

document.addEventListener('DOMContentLoaded', () => {
    initBengaluruMap();
    initBengaluruCharts();
    initBengaluruFilters();
    initBengaluruPrediction();
});

function initBengaluruMap() {
    const mapEl = document.getElementById('bengaluruMap');
    if (!mapEl) return;

    // Center on Bengaluru Central
    bMap = L.map('bengaluruMap').setView([12.9716, 77.5946], 11);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(bMap);

    wardLayerGroup = L.layerGroup().addTo(bMap);
    alertLayerGroup = L.layerGroup().addTo(bMap);

    const initialHotspots = window.bengaluruData?.hotspots || [];
    const initialAlerts = window.bengaluruData?.sampleAlerts || [];

    renderHotspotMarkers(initialHotspots);
    renderAlertMarkers(initialAlerts.slice(0, 300));
}

function renderHotspotMarkers(hotspots) {
    if (!wardLayerGroup) return;
    wardLayerGroup.clearLayers();

    hotspots.forEach(h => {
        if (!h.lat || !h.lon) return;

        const isHigh = h.Risk_Rating === 'High Risk';
        const color = isHigh ? '#ef4444' : '#f59e0b';
        const radius = Math.min(Math.max(Math.sqrt(h.total_alerts || 100) * 0.4, 7), 22);

        const circle = L.circleMarker([h.lat, h.lon], {
            radius: radius,
            fillColor: color,
            color: '#ffffff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        });

        const popupContent = `
            <div style="font-size: 0.85rem; min-width: 180px;">
                <div class="fw-bold text-slate-900 mb-1" style="font-size: 0.95rem;">${h.Area}</div>
                <div class="mb-1"><span class="badge ${isHigh ? 'bg-danger' : 'bg-warning text-dark'}">${h.Risk_Rating}</span></div>
                <hr class="my-1">
                <div><strong>Total CAS Alerts:</strong> ${(h.total_alerts || 0).toLocaleString()}</div>
                <div><strong>High-Risk Events:</strong> ${(h.high_risk_alerts || 0).toLocaleString()}</div>
                <div><strong>Accident Severity:</strong> ${h.Accident_Severity || 'Medium'}</div>
                <div><strong>Pothole Severity:</strong> ${h.Pothole_Severity || 'Medium'}</div>
                <div><strong>Top Warning:</strong> ${h.top_alarm || 'N/A'}</div>
            </div>
        `;

        circle.bindPopup(popupContent);

        circle.on('click', () => {
            showWardDetail(h);
        });

        wardLayerGroup.addLayer(circle);
    });

    const statusBadge = document.getElementById('mapStatusBadge');
    if (statusBadge) {
        statusBadge.textContent = `Rendering ${hotspots.length} active wards`;
    }
}

function showWardDetail(h) {
    const placeholder = document.getElementById('wardDetailPlaceholder');
    const content = document.getElementById('wardDetailContent');
    if (!placeholder || !content) return;

    placeholder.style.display = 'none';
    content.style.display = 'block';

    document.getElementById('detailWardName').textContent = h.Area || 'Unknown';
    const badge = document.getElementById('detailRiskBadge');
    badge.textContent = h.Risk_Rating || 'Moderate Risk';
    badge.className = `badge ${h.Risk_Rating === 'High Risk' ? 'bg-danger' : 'bg-warning text-dark'}`;

    document.getElementById('detailAlerts').textContent = (h.total_alerts || 0).toLocaleString();
    document.getElementById('detailAccident').textContent = h.Accident_Severity || 'Medium';
    document.getElementById('detailPothole').textContent = h.Pothole_Severity || 'Medium';
    document.getElementById('detailSpeed').textContent = `${h.avg_speed || 0} km/h`;
    document.getElementById('detailTopAlarm').textContent = h.top_alarm || 'PCW';
}

function renderAlertMarkers(alerts) {
    if (!alertLayerGroup) return;
    alertLayerGroup.clearLayers();

    alerts.forEach(a => {
        if (!a.lat || !a.lon) return;

        const circle = L.circleMarker([a.lat, a.lon], {
            radius: 3,
            fillColor: '#2563eb',
            color: '#1d4ed8',
            weight: 1,
            opacity: 0.8,
            fillOpacity: 0.6
        });

        circle.bindPopup(`
            <div style="font-size: 0.8rem;">
                <strong>Ward:</strong> ${a.ward}<br>
                <strong>Warning:</strong> ${a.alarm}<br>
                <strong>Speed:</strong> ${a.speed || 0} km/h<br>
                <strong>Timestamp:</strong> ${a.time ? a.time.replace('T', ' ').substring(0, 19) : 'N/A'}
            </div>
        `);

        alertLayerGroup.addLayer(circle);
    });
}

function initBengaluruCharts() {
    const dist = window.bengaluruData?.alarmDistribution || {};
    const alarmCtx = document.getElementById('alarmChart');

    if (alarmCtx && Object.keys(dist).length > 0) {
        new Chart(alarmCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(dist),
                datasets: [{
                    label: 'Sample Alert Count',
                    data: Object.values(dist),
                    backgroundColor: [
                        '#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#64748b'
                    ],
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
}

function initBengaluruFilters() {
    const applyBtn = document.getElementById('bengaluruFilterBtn') || document.getElementById('applyFiltersBtn');
    const resetBtn = document.getElementById('bengaluruResetBtn') || document.getElementById('resetFiltersBtn');

    if (!applyBtn) return;

    applyBtn.addEventListener('click', async () => {
        const risk = document.getElementById('filterRisk').value;
        const accident = document.getElementById('filterAccident').value;
        const pothole = document.getElementById('filterPothole').value;

        applyBtn.disabled = true;
        applyBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Filtering...';

        try {
            const url = `/api/bengaluru/filter?risk=${encodeURIComponent(risk)}&accident_severity=${encodeURIComponent(accident)}&pothole_severity=${encodeURIComponent(pothole)}`;
            const resp = await fetch(url);
            const data = await resp.json();

            if (data.error) {
                showToast(data.error, 'danger');
                return;
            }

            renderHotspotMarkers(data.hotspots);
            renderAlertMarkers(data.sample_alerts || []);
            updateHotspotTable(data.hotspots);

            showToast(`Filter applied: ${data.count} wards match criteria`, 'success');
        } catch (err) {
            console.error(err);
            showToast('Failed to apply filters.', 'danger');
        } finally {
            applyBtn.disabled = false;
            applyBtn.innerHTML = '<i class="bi bi-funnel me-1"></i> Apply Filter';
        }
    });

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            document.getElementById('filterRisk').value = 'All';
            document.getElementById('filterAccident').value = 'All';
            document.getElementById('filterPothole').value = 'All';
            applyBtn.click();
        });
    }
}

function initBengaluruPrediction() {
    const form = document.getElementById('bengaluruPredictForm');
    const btn = document.getElementById('blrPredictBtn');
    const placeholder = document.getElementById('blrResultsPlaceholder');
    const content = document.getElementById('blrResultsContent');

    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Evaluating Hotspot...';

        const payload = {
            area: document.getElementById('blr_area').value,
            alarm_type: document.getElementById('blr_alarm').value,
            speed: parseFloat(document.getElementById('blr_speed').value) || 45,
            time_cat: document.getElementById('blr_time').value,
            condition: document.getElementById('blr_condition').value,
            visibility: document.getElementById('blr_visibility').value
        };

        try {
            const resp = await fetch('/api/predict/bengaluru', {
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

            // Badge & Score
            const badge = document.getElementById('blrResBadge');
            badge.textContent = data.prediction;
            badge.className = `badge px-3 py-2 fs-6 ${data.badge_class}`;

            document.getElementById('blrResScoreText').textContent = `Risk Score: ${data.risk_score} / 100`;

            // Probabilities
            if (data.probabilities) {
                const highP = data.probabilities['High Risk'] || 0;
                const lowP = data.probabilities['Low Risk'] || 0;

                document.getElementById('blrProbHigh').textContent = `${highP}%`;
                document.getElementById('blrBarHigh').style.width = `${highP}%`;

                document.getElementById('blrProbLow').textContent = `${lowP}%`;
                document.getElementById('blrBarLow').style.width = `${lowP}%`;
            }

            // Risk Triggers
            const rfList = document.getElementById('blrRiskFactorsList');
            rfList.innerHTML = '';
            (data.risk_factors || []).forEach(f => {
                const li = document.createElement('li');
                li.textContent = f;
                rfList.appendChild(li);
            });

            // Suggestions
            const suggList = document.getElementById('blrSuggestionsList');
            suggList.innerHTML = '';
            (data.suggestions || []).forEach(s => {
                const li = document.createElement('li');
                li.textContent = s;
                suggList.appendChild(li);
            });

            // Advisory
            document.getElementById('blrResAdvisory').textContent = data.advisory || 'Drive defensively.';

            showToast(`Predicted: ${data.prediction} (${data.risk_score}/100)`, 'success');

        } catch (err) {
            console.error(err);
            showToast('Failed to evaluate Bengaluru hotspot risk.', 'danger');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-lightning-charge me-1"></i> Calculate Hotspot Status & Directives';
        }
    });
}

function updateHotspotTable(hotspots) {
    const tbody = document.getElementById('hotspotsTableBody');
    if (!tbody) return;

    tbody.innerHTML = '';
    if (hotspots.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-3">No wards match the selected criteria.</td></tr>';
        return;
    }

    hotspots.forEach(h => {
        const tr = document.createElement('tr');
        const badgeClass = h.Risk_Rating === 'High Risk' ? 'bg-danger' : 'bg-warning text-dark';
        tr.innerHTML = `
            <td class="fw-semibold text-slate-900">${h.Area}</td>
            <td>${(h.total_alerts || 0).toLocaleString()}</td>
            <td><span class="badge ${badgeClass}">${h.Risk_Rating}</span></td>
            <td>${h.Accident_Severity}</td>
            <td>${h.Pothole_Severity}</td>
        `;
        tbody.appendChild(tr);
    });
}
