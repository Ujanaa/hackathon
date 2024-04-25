var map;
var lastZoomLevel = 0;
var latitudes;
var longitudes;
var fireDetected = false;

        function GetMap() {
            map = new Microsoft.Maps.Map('#myMap', {});
            Microsoft.Maps.Events.addHandler(map, 'viewchangeend', function () { set_latitudes_and_longitude(map); });
            lastZoomLevel = map.getZoom();
        }        
        function set_latitudes_and_longitude(map)
        {
            latitudes = map._v8Map._view.cameraLocation.latitude
            longitudes = map._v8Map._view.cameraLocation.longitude
          
            if(lastZoomLevel != map.getZoom()){
               lastZoomLevel = map.getZoom();
            }
            
        }
        async function getscanned(){
            try{
                const response = await axios.get(`https://firedet-production.up.railway.app/api/${latitudes}/${longitudes}/${lastZoomLevel}`)
                console.log(response.data.data["0"])
                if(response.data.data["0"] > 0.5) fireDetected = true;
                else fireDetected = false
                var result = document.getElementById("result");
                if(fireDetected){
                    result.innerText = "Fire Detected";
                    result.style.color = "red";
                }
                else{
                    console.log(result)
                    result.innerText = "No fire detected" ; result.style.color = "green";
                }
                
                // console.log(latitudes , longitudes , lastZoomLevel)
            }
            catch (err) {
            alert("something went wrong. Try again");
          }
        }
