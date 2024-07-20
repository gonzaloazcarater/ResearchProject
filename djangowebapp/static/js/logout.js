// logout.js

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('logout-form').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission
        
        // Send a POST request to the logout endpoint
        fetch('/logout', {  // Make sure the route is correct
            method: 'POST',
            headers: {
                'X-CSRFToken': getCookie('csrftoken') // Get the CSRF token
            }
        })
        .then(function(response) {
            if (response.ok) {
                // If the request is successful, redirect to the login page
                window.location.replace('/login'); // Make sure the route is correct
            } else {
                // If there's an issue with the request, log the error to the console
                console.error('Error logging out:', response.statusText);
            }
        })
        .catch(function(error) {
            // If there's an error with the request, log the error to the console
            console.error('Error:', error);
        });
    });
});

// Function to get the CSRF token value
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            // Find the CSRF token
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
