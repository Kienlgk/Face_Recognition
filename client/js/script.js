$(document).on("click", ".btn-get-stream", function(){
    fetchdata();
});

function fetchdata(){
//  $.ajax({
//   url: 'http://127.0.0.1:8888/facerecognitionLive',
//   type: 'GET',
//   dataType: 'jsonp',
//   crossDomain: true,
// //   crossOrigin: true,
//   success: function(res){
//     print(res);
//     // Perform operation on the return value
//     res = JSON.parse(res)
//     base64_string = res.img
//     print(base64_string)
//     $('.streaming').setAttribute('src', 'data:image/jpg;base64,' + base64_string);
//   }
//  });
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
        decoded_file = atob(res.img)
        blob = new Blob([decoded_file], {type: 'application/octet-stream'});
        var imageUrl = URL.createObjectURL(blob)
        
        $('.streaming').attr('src', imageUrl);
        // urlCreatorrevokeObjectURL()
        // var title = getTitle(text);
        // alert('Response from CORS request to ' + url + ': ' + title);
    };

    xhr.onerror = function() {
        alert('Woops, there was an error making the request.');
    };

    xhr.send();
}

//$(document).ready(function(){
// setInterval(fetchdata,5000);
//});