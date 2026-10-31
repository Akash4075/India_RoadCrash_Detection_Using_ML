// Shared UI Utilities for Road Safety Intelligence System

function showToast(message, type = 'primary') {
    const toastEl = document.getElementById('appToast');
    const toastMsg = document.getElementById('toastMessage');
    if (!toastEl || !toastMsg) return;

    toastEl.className = `toast align-items-center text-bg-${type} border-0 shadow`;
    toastMsg.textContent = message;

    const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
    toast.show();
}

document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips if Bootstrap tooltip is available
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
