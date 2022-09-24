import ee

ee.Initialize()

dataset = ee.ImageCollection('USDA/NAIP/DOQQ').filter(ee.Filter.date('2019-01-01', '2020-12-31'));

llx = -74.0
lly = 41.0
urx = -75
ury = 42.0
geometry = [[llx,lly], [llx,ury], [urx,ury], [urx,lly]]

task = ee.batch.Export.image.toDrive(image=dataset.mean(),
                                     scale=0.5,
                                     region=geometry,
                                     fileNamePrefix='my_export',
                                     crs='EPSG:3067',
                                     fileFormat='GEO_TIFF')
task.start()