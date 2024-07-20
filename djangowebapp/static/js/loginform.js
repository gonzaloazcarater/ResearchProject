function togglePasswordVisibility() {
    var passwordField = document.getElementById("passwordfield");
    if (passwordField.type === "password") {
        passwordField.type = "text";
    } else {
        passwordField.type = "password";
    }
}

document.addEventListener("DOMContentLoaded", function() {
    var showPasswordToggle = document.querySelector(".showPasswordToggle");
    if (showPasswordToggle) {
        showPasswordToggle.addEventListener("click", togglePasswordVisibility);
    }
});
