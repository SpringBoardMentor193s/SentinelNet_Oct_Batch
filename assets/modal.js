// openModal(id) and closeModal(id)
function openModal(id) {
  const overlay = document.getElementById('cyber-overlay');
  const modal = document.getElementById(id);
  if (overlay && modal) {
    overlay.classList.add('show');
    modal.classList.add('show');
  }
}
function closeModal(id) {
  const overlay = document.getElementById('cyber-overlay');
  const modal = document.getElementById(id);
  if (overlay && modal) {
    overlay.classList.remove('show');
    modal.classList.remove('show');
  }
}
