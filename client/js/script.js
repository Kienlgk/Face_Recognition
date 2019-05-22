$(document).on("click", ".btn-get-stream", function(){
    fetchdata();
});

function fetchdata(){
 $.ajax({
  url: 'localhost:8888/facerecognitionLive',
  type: 'get',
  success: function(base64_endcode){
   // Perform operation on the return value
   base64_string = base64_endcode
   $('.streaming').setAttribute('src', 'data:image/jpg;base64,' + base64_string);
  }
 });
}

//$(document).ready(function(){
// setInterval(fetchdata,5000);
//});