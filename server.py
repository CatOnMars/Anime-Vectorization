from bottle import route, run, request, template, response, static_file
import os
import cv2 as cv
import pytorchWrapper as model

@route('/vectorize', method='POST')
def do_upload():
    upload     = request.files.get('file')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'

    save_path = "WebUI/uploads/"
    upload.save(save_path,overwrite=True) # appends upload.filename automatically
    #img=cv.imread(save_path+upload.filename)
    #img=cv.imencode('.png',img)
    #response.set_header('Content-type', 'image/png')
    model.getEdgeMap(upload)
    return "OK"


@route('/edgeMap', method='POST')
def do_upload():
    upload     = request.files.get('file')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'
    print(upload.filename)
    #save_path = "WebUI/uploads/"
    #upload.save(save_path,overwrite=True) # appends upload.filename automatically
    #img=cv.imread(save_path+upload.filename)
    #img=cv.imencode('.png',img)
    #response.set_header('Content-type', 'image/png')
    res=model.setEdgeMap(upload)
    if res:
        return "OK"
    else:
        response.status=404
        return "NotYetComplete"

@route('/csMap', method='POST')
def do_upload():
    upload     = request.files.get('file')
    name, ext = os.path.splitext(upload.filename)
    print(upload.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'

    #save_path = "WebUI/uploads/"
    #upload.save(save_path,overwrite=True) # appends upload.filename automatically
    #img=cv.imread(save_path+upload.filename)
    #img=cv.imencode('.png',img)
    #response.set_header('Content-type', 'image/png')
    res=model.setCsMap(upload)
    if res:
        return "OK"
    else:
        response.status=404
        return "NotYetComplete"

@route('/')
def hello():
    return static_file('index.html',root='WebUI/')

@route('/uploads/<filename>')
def send_image(filename):
    if filename.endswith('png'):
        return static_file(filename, root='WebUI/uploads/', mimetype='image/png')
    elif filename.endswith('jpg'):
        return static_file(filename, root='WebUI/uploads/', mimetype='image/jpeg')

@route('/edgeMap/<filename>')
def send_image(filename):
    if filename.endswith('png'):
        return static_file(filename, root='WebUI/edgeMap/', mimetype='image/png')
    elif filename.endswith('jpg'):
        return static_file(filename, root='WebUI/edgeMap/', mimetype='image/jpeg')

@route('/bEdgeMap/<filename>')
def send_image(filename):
    if filename.endswith('png'):
        return static_file(filename, root='WebUI/bEdgeMap/', mimetype='image/png')
    elif filename.endswith('jpg'):
        return static_file(filename, root='WebUI/bEdgeMap/', mimetype='image/jpeg')

@route('/reconstruction/<filename>')
def send_image(filename):
    if filename.endswith('png'):
        return static_file(filename, root='WebUI/reconstruction/', mimetype='image/png')
    elif filename.endswith('jpg'):
        return static_file(filename, root='WebUI/reconstruction/', mimetype='image/jpeg')


@route('/curves/<filename>')
def send_json(filename):
    return static_file(filename.rstrip('.jpg').rstrip(".png")+".json",root="WebUI/json/",mimetype="json")

run(host='localhost', port=8080)