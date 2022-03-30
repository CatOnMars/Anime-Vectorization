## Goal
This is a prototypical user interface for converting images to vectorized format.<br>
See a demo at: https://youtu.be/Ge4ZJ5FiHmE <br>
The application part start from 3:03.

This project is still in development;<br>
all non-opensource part are excluded out of this repository.<br> 

## Manifest
__server.py__ - APIs are defined here. <br>
__pytorchWrapper.py__ - a wrapper for dealing with pyTorch models. <br>
__WebUI/__ - the web user interface frontend.<br>
__pyVec/__ - python C extention for fast curve extraction.<br>
__PG2020_DDC.pdf__ - Paper draft.<br>

## Architecture
This system consists of two parts: the web page UI and the API server.
The web UI was built using Vue.js and the API server was built using 
bottle.py, pyTorch, and a custom python extension "pyVec".
In the development environment, no database involved.

The web UI implement the curve editing tool and sent the curve data 
to the server for rendering. The server renders the image and send back 
image data for the web UI to display. 

The web UI's curve editing tool was built for <canvas> with a '2d' context.

The curve data is stored in JSON objects:
	
~~~
	{
		"curve_count": 1012,
		"curves": [
			{
				"bestScale": [
					1,
					1,
					1
				],
				"bs_size": 2,
				"lc_size": 2,
				"leftColor": [
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1,
					1
				],
				"pl_size": 3,
				"polyline_coord": [
					7,
					1,
					7,
					128,
					7,
					509
				],
				"rc_size": 2,
				"rightColor": [
					254,
					254,
					254,
					254,
					254,
					254,
					254,
					254,
					254
				]
			},
			{
				...
			}
		]
	}
~~~
					
The server-side API is defined as:

>__POST /vectorize__ : the request contain an jpg or png image, this endpoint do the vectorization.<br>
>		This API returns "OK" when the vectorized data (curve JSON, edgeMap, colorSourceMap) are ready.
>
>__POST /edgeMap__ :<br>
>__POST /csMap__ :    Since the input to our neural network are two images, the API user could upload edgeMap and colorSourceMap 
>for the server to do rendering. Note that the rendered image are ready while BOTH API returned "OK".<br>
>
>__GET /reconstruction/<filename>__ : The rendered image. <br>
>__GET /curves/<filename>__ : The vectorized curve JSON data.<br>


## Discussion
Current API have the problem that if two user upload files that have the same name, they may corrupt each other's data.
The solution is simple: change the API to /curves/<username>/<filename> where <username> is unique.

The curve polyline editing tool could be optimized. Currently the control point picking algorithm is brute-force search.
Spatial data structure such as Quad Tree and regular-sized grid can be used to accelerate the spatial query.

The web UI's curve editing tool is built for <canvas> with a '2d' context. Alternatively, an WebGL context could be used
for better performance.

This architecture is far from optimal, 
the rendering and vectorization time is about 30 ms in the dev environment, 
however, the data-packing/unpacking and tranmission overhead is around 150 ms even the server and WebUI are running 
on the same machine. A better architecture would be to store the neural network model on the client side, and 
do all the vectorization/rendering in the browser. The pytorch model could be converted to ONNX format and using 
deep learning framework for the web browser.
