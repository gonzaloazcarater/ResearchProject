// scripts.js

document.addEventListener('DOMContentLoaded', function() {
    var getStartedBtn = document.getElementById('getStartedBtn');
    setTimeout(function() {
        getStartedBtn.style.opacity = 1;  // Hacer el botón visible
        getStartedBtn.classList.add('slide-in');  // Agregar la clase para la animación
    }, 1000);  // 1000 milisegundos = 1 segundos
});

