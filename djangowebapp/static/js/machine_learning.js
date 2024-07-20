function handleSubmit() {
    const modelSelect = document.getElementById('model_select').value;
    let targetUrl = '';
    switch(modelSelect) {
        case 'knn':
            targetUrl = urls.knn;
            break;
        case 'decision_tree':
            targetUrl = urls.decision_tree;
            break;
        case 'linear_regression':
            targetUrl = urls.linear_regression;
            break;
        case 'svm':
            targetUrl = urls.svm;
            break;
        case 'summary':
            targetUrl = urls.summary;
            break;
        default:
            alert('Please select a model.');
            return;
    }
    window.location.href = targetUrl;
}
