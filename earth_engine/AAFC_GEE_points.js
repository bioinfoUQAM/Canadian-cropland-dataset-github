// Create a crop landcover type dataset using the Google Earth Engine 

// Retrieve the most recent image from the Canadian Annual Crop Inventory (ACI). 
// There is a single image per year for this dataset.
var ACI = ee.ImageCollection("AAFC/ACI");
var crop2019 = ACI.filter(ee.Filter.date('2019-01-01', '2019-12-31')).first()
print(crop2019);
print(ACI);
Map.addLayer(crop2019, { }, "Complete Annual Crop Inventory");

// Get some basic info from the Crop Inventory 

// Get information about the bands as a list.
var bandNames = crop2019.bandNames();
print('Band names:', bandNames); 
// Retrieve the scale of the images 
var b1scale = crop2019.select('landcover').projection().nominalScale();
print('Band 1 scale:', b1scale);  // ee.Number 
// Get a list of all metadata properties.
var properties = crop2019.propertyNames();
print('Metadata properties:', properties);  // ee.List of metadata properties
// Number of images in the ACI dataset
print("Number of images in collection", ACI.size());
print("ACI 2019 Info", crop2019)

// Create a list of the crops of interest we would like to look at for this dataset and their band values. 
var crops = {
  "Spring Wheat": 146,
  "Soybeans": 158, 
  "Corn": 147, 
  "Potatoes": 177, 
  "Oats": 136, 
  "Millet": 135, 
  "Barley": 133, 
  "Pasture and Forages": 122, 
  "Mixedwood": 230
}

// Create a function to filter the cloud cover, ice and shadow data 
function maskCloudAndShadows(image) {
var cloudProb = image.select('MSK_CLDPRB');
var snowProb = image.select('MSK_SNWPRB');
// Apply some (strictly) less than filters to the image  
var cloud = cloudProb.lt(8); 
var snow = snowProb.lt(8);
var scl = image.select('SCL');
var shadow = scl.eq(3); // 3 = cloud shadow
var cirrus = scl.eq(10); // 10 = cirrus
var mask = (cloud.and(snow)).and(cirrus.neq(1)).and(shadow.neq(1));
return image.updateMask(mask);
}

// Apply filter where country name equals Canada. 
var CanadaBorder = table.filter(ee.Filter.eq('country_na', 'Canada'));

var vizParams = {bands: ['B4', 'B3', 'B2'], min: 0, max: 2000};

// Retrieve some imagery from Sentinel-2. Use Sentinel-2 L2A data - which has better cloud masking

// JUNE 2019
var SEN2 = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2019-06-01', '2019-06-30').filterBounds(CanadaBorder).map(maskCloudAndShadows);
Map.addLayer(SEN2.median(), vizParams, 'Sentinel-2 Imagery (June 2019)');

// JULY 2019
var SEN2 = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2019-07-01', '2019-07-31').filterBounds(CanadaBorder).map(maskCloudAndShadows);
Map.addLayer(SEN2.median(), vizParams, 'Sentinel-2 Imagery (July 2019)');

// AUGUST 2019
var SEN2 = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2019-08-01', '2019-08-31').filterBounds(CanadaBorder).map(maskCloudAndShadows);
Map.addLayer(SEN2.median(), vizParams, 'Sentinel-2 Imagery (August 2019)');

// OCTOBER 2019
var SEN2 = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2019-09-01', '2019-09-30').filterBounds(CanadaBorder).map(maskCloudAndShadows);
Map.addLayer(SEN2.median(), vizParams, 'Sentinel-2 Imagery (September 2019)');

var SEN2 = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2019-10-01', '2019-10-31').filterBounds(CanadaBorder).map(maskCloudAndShadows);
Map.addLayer(SEN2.median(), vizParams, 'Sentinel-2 Imagery (October 2019)');


// Loop through the various crops types and extract the particular layer 
for (var key in crops) {
  var crop_mask = crop2019.select('landcover').eq(crops[key]); // Select a specific crop
  var crop = crop2019.updateMask(crop_mask.not());
  crop = crop.cast({"landcover": "double"})
  Map.addLayer(crop, {palette: '000000'}, key);  //Black-out non feature
}


// Add a layer for the province boundaries 
var provinces = ee.FeatureCollection("FAO/GAUL/2015/level1");

var styleParams = {
  fillColor: '00000000',
  color: '00909F',
  width: 1.0,
};

provinces = provinces.style(styleParams);
Map.addLayer(provinces, {}, 'First Level Administrative Units');
