
function fetchdata(){
    makeCorsRequest();
}

// Create the XHR object.
function createCORSRequest(method, url) {
    var xhr = new XMLHttpRequest();
    if ("withCredentials" in xhr) {
      // XHR for Chrome/Firefox/Opera/Safari.
      xhr.open(method, url, true);
    } else if (typeof XDomainRequest != "undefined") {
      // XDomainRequest for IE.
      xhr = new XDomainRequest();
      xhr.open(method, url);
    } else {
      // CORS not supported.
      xhr = null;
    }
    return xhr;
  }
  
// Make the actual CORS request.
function makeCorsRequest() {
    // This is a sample server that supports CORS.
    var url = 'http://localhost:8888/facerecognitionLive';

    var xhr = createCORSRequest('GET', url);
    if (!xhr) {
        alert('CORS not supported');
        return;
    }

    // Response handlers.
    xhr.onload = function() {
        var jsonRes = xhr.responseText;
        res = JSON.parse(jsonRes)
        console.log(res.img)

        $('.streaming').attr('src', 'data:image/gif;base64,'+res.img);
    };

    xhr.onerror = function() {
        alert('Woops, there was an error making the request.');
    };

    xhr.send();
}

$(document).ready(function(){
    $(document).on("click", ".btn-get-stream", function(){
        fetchdata();
    });
    setInterval(fetchdata, 1000);
});