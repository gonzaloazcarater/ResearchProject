function validatePasswords() {
    var password = document.getElementById("passwordfield").value;
    var confirmPassword = document.getElementById("confirmpasswordfield").value;
    if (password !== confirmPassword) {
        document.getElementById("passwordError").style.display = "block";
        return false;
    }
    document.getElementById("passwordError").style.display = "none";
    return true;
}

function togglePasswordVisibility(fieldId, toggleElement) {
    var passwordField = document.getElementById(fieldId);
    if (passwordField.type === "password") {
        passwordField.type = "text";
        toggleElement.textContent = "HIDE";
    } else {
        passwordField.type = "password";
        toggleElement.textContent = "SHOW";
    }
}
