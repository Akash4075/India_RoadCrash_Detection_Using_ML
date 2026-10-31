// India Road Accident Risk Analysis Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
    initIndiaCharts();
    initIndiaForm();
});

function initIndiaCharts() {
    const stats = window.indiaStats || {};

    // 1. Severity Distribution
    const sevCtx = document.getElementById('severityChart');
    if (sevCtx && stats.severityCounts) {
        new Chart(sevCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(stats.severityCounts),
                datasets: [{
                    data: Object.values(stats.severityCounts),
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
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

    // 2. Day vs Night Distribution
    const dnCtx = document.getElementById('dayNightChart');
    if (dnCtx && stats.dayNight) {
        new Chart(dnCtx, {
            type: 'pie',
            data: {
                labels: Object.keys(stats.dayNight),
                datasets: [{
                    data: Object.values(stats.dayNight),
                    backgroundColor: ['#38bdf8', '#1e293b'],
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

    // 3. Top States
    const stateCtx = document.getElementById('topStatesChart');
    if (stateCtx && stats.topStates) {
        new Chart(stateCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(stats.topStates),
                datasets: [{
                    label: 'Accidents Recorded',
                    data: Object.values(stats.topStates),
                    backgroundColor: '#2563eb',
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

function initIndiaForm() {
    const form = document.getElementById('indiaPredictForm');
    const btn = document.getElementById('predictBtn');
    const placeholder = document.getElementById('resultsPlaceholder');
    const content = document.getElementById('resultsContent');

    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span> Calculating Risk...';

        const payload = {
            weather: document.getElementById('weather').value,
            road_condition: document.getElementById('road_condition').value,
            lighting: document.getElementById('lighting').value,
            traffic: document.getElementById('traffic').value,
            vehicle: document.getElementById('vehicle').value,
            speed_limit: parseFloat(document.getElementById('speed_limit').value) || 60
        };

        try {
            const resp = await fetch('/api/predict/india', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await resp.json();

            if (data.error) {
                showToast(data.error, 'danger');
                return;
            }

            // Display Results
            placeholder.style.display = 'none';
            content.style.display = 'block';

            // Risk Tier Badge and Score
            const badge = document.getElementById('resBadge');
            badge.textContent = data.prediction || data.risk_tier || 'Moderate Risk';
            badge.className = `badge px-3 py-2 fs-6 ${data.badge_class || 'badge-warning'}`;

            const scoreText = document.getElementById('resScoreText');
            if (scoreText && data.risk_score !== undefined) {
                scoreText.textContent = `Risk Index: ${data.risk_score} / 100`;
            }

            // Severity Probability bars
            if (data.probabilities) {
                const minor = data.probabilities.Minor || 0;
                const serious = data.probabilities.Serious || 0;
                const fatal = data.probabilities.Fatal || 0;

                document.getElementById('probMinor').textContent = `${minor}%`;
                document.getElementById('barMinor').style.width = `${minor}%`;

                document.getElementById('probSerious').textContent = `${serious}%`;
                document.getElementById('barSerious').style.width = `${serious}%`;

                document.getElementById('probFatal').textContent = `${fatal}%`;
                document.getElementById('barFatal').style.width = `${fatal}%`;
            }

            // Risk Factors Identified
            const rfList = document.getElementById('riskFactorsList');
            rfList.innerHTML = '';
            if (data.risk_factors && data.risk_factors.length > 0) {
                data.risk_factors.forEach(f => {
                    const li = document.createElement('li');
                    li.textContent = f;
                    rfList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'Favorable driving conditions detected across all parameters.';
                rfList.appendChild(li);
            }

            // Tailored Preventive Suggestions
            const suggList = document.getElementById('suggestionsList');
            if (suggList) {
                suggList.innerHTML = '';
                if (data.suggestions && data.suggestions.length > 0) {
                    data.suggestions.forEach(s => {
                        const li = document.createElement('li');
                        li.textContent = s;
                        suggList.appendChild(li);
                    });
                }
            }

            // Master Executive Advisory
            const advBox = document.getElementById('resAdvisory');
            if (advBox) {
                advBox.textContent = data.advisory || 'Drive with heightened situational awareness and observe speed limits.';
            }

            showToast(`Predicted: ${data.prediction} (${data.risk_score}/100)`, 'success');

        } catch (err) {
            console.error(err);
            showToast('Failed to connect to the prediction API.', 'danger');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-shield-check me-1"></i> Predict Accident Risk & Generate Advice';
        }
    });
}
