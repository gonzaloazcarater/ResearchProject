$(document).ready(function() {
    $('.toggle-favorite-form').on('submit', function(e) {
        e.preventDefault(); // Prevents the default form submission behavior
        console.log('Form submitted but redirection prevented');
        
        var $form = $(this);
        var $button = $form.find('.favorite');
        var isFavorite = $button.hasClass('selected');
        
        // Update the value of is_favorite before submitting the form
        $form.find('input[name="is_favorite"]').val(!isFavorite);

        $.ajax({
            url: $form.attr('action'),
            method: 'POST',
            data: $form.serialize(),
            success: function(response) {
                console.log('Server response:', response);
                if (response.success) {
                    // Toggle the visual state of the star button
                    $button.toggleClass('selected');
                } else {
                    console.log('Error marking as favorite');
                }
            },
            error: function(xhr, errmsg, err) {
                console.log('AJAX request error');
            }
        });
    });
});
