#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
//#undef slots
//#include <Python.h>
//#define slots
#include "edge_extract_reconstruct/multiscale_image.h"
#include "edge_extract_reconstruct/image_tools.h"
//#include "edge_extract_reconstruct/solver.h"
#include "edge_extract_reconstruct/blur.h"

#include "contour_fit/contour.h"
#include "contour_fit/curve_set.h"
#include "contour_fit/junction_extraction.h"

#include "global_param.h"

#include <iostream>


#include <QGuiApplication>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QFile>

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input) {

	if (input.ndim() != 3)
		throw std::runtime_error("3-channel image must be 3 dims ");

	py::buffer_info buf = input.request();

	cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

	return mat;
}

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat& input) {

	py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows,input.cols,3 }, input.data);
	return dst;
}


void save_myimage(IplImage* image, const char* imageName, const char*  directory)
{
	char dir[1024];
	strcpy_s(dir, directory);
	strcat_s(dir, imageName);
	strcat_s(dir, ".png");

	int width = image->width;
	int height = image->height;
	IplImage* img_tmp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	CvScalar p;

	for (int i = 0; i<height; i++) {
		for (int j = 0; j<width; j++) {
			p = cvGet2D(image, i, j);
			cvSet2D(img_tmp, i, j, p);
		}
	}

	if (!cvSaveImage(dir, img_tmp)) std::cerr << "Could not save " << imageName << "\n";
	cvReleaseImage(&img_tmp);
	delete img_tmp;
	return;
}

py::list trackContour(py::array_t<unsigned char>& edgeMap_numpy, py::array_t<unsigned char>& originalImg_numpy, char* xml_filename) {
	long t1 = cv::getTickCount();
	cv::Mat tmp1 = numpy_uint8_3c_to_cv_mat(edgeMap_numpy);
	cv::Mat tmp2 = numpy_uint8_3c_to_cv_mat(originalImg_numpy);
	IplImage edgeMap=IplImage(tmp1);
	IplImage original_image=IplImage(tmp2);

	//convert the image to floating point
	IplImage* imgf = cvCreateImage(cvSize(edgeMap.width, edgeMap.height), IPL_DEPTH_32F, 3);
	cvZero(imgf);
	CvScalar pt;
	for (int i = 0; i<edgeMap.height; i++) {
		for (int j = 0; j<edgeMap.width; j++) {
			pt = cvGet2D(&edgeMap, i, j);
	//		if (j == 83|| j==82 ||j==84) {
	//			cout<<"SHould trace:"<<pt.val[0]<<pt.val[1]<<pt.val[2]<<pt.val[3] <<endl;
	//		}
			if (pt.val[0] >= 10) {
				//convert bestScale to deepness
				//		int deepness=ceil(((pt.val[0] + pt.val[1] + pt.val[2]) / 3.0f)/255.0*10);
				//		pt.val[0] = pt.val[1] = pt.val[2] = deepness;
				cvSet2D(imgf, i, j, pt);
			}
		}
	}
/*
	cv::Mat imgff = cv::cvarrToMat(imgf);

	int erosion_elem = 0;
	int erosion_size = 0;
	int dilation_elem = 0;
	int dilation_size = 1;
	int const max_elem = 2;
	int const max_kernel_size = 21;

	int dilation_type = 0;
	if (dilation_elem == 0) { dilation_type = cv::MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = cv::MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }

	cv::Mat element = cv::getStructuringElement(dilation_type,
		cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		cv::Point(dilation_size, dilation_size));
	cv::dilate(imgff,imgff, element);
	cv::erode(imgff, imgff, element);*/
	IplImage* edges = imgf;
	save_myimage(edges, "edgeMap", "WebUI/");

	//release temporary image
	//cvReleaseImage(&edgeMap);
	int width = imgf->width;
	int height = imgf->height;


	long t2 = cv::getTickCount();
	std::vector<Contour> contour_paths;
	CurveSet curve_set = CurveSet(width, height);
	std::cout << "Track pixel chains ....\n";
	size_t size_track = track_pixel_chains(edges, contour_paths);
	std::cout << "Chains found: " << size_track << "....\n";

	long t3 = cv::getTickCount();
	size_track = contour_paths.size();

	for (size_t chain_nb = 0; chain_nb<size_track; chain_nb++)
	{
		//vectorize the pixels, don't touch the attributes
		contour_paths[chain_nb].remove_duplicate_pixels();
		bool lengthOk = contour_paths[chain_nb].fill_bezier_approximation();
		if (!lengthOk)
		{
			std::cout << "Chain with length 0: " << chain_nb << " with size " << contour_paths[chain_nb].getPixelChainSize() << "\n";
			contour_paths.erase(contour_paths.begin() + chain_nb);
			size_track = contour_paths.size();
			chain_nb--;
		}
		else {
			//fills and smooths the bestScale
			//fills and averages the lifetime
			//uses the smoothed version of the bestScale to fill the colors
			bool colorOk = contour_paths[chain_nb].fill_pixel_data(edges, &original_image);
			//	bool colorOk = contour_paths[chain_nb].fill_pixel_data(edges, original_image, bestScale, gx, gy);
			//if color is -1 everywhere, erase the chain
			if (!colorOk)
			{
				std::cout << "Chain with color null: " << chain_nb << " with size " << contour_paths[chain_nb].getPixelChainSize() << "\n";
				contour_paths.erase(contour_paths.begin() + chain_nb);
				size_track = contour_paths.size();
				chain_nb--;
			}
		}
	}


	//cvReleaseImage(&edges);
	//cvReleaseImage(&original_image);

	//delete edges;
	//delete original_image;
	//delete color_position;
	//delete color;
	//delete color_L;
	//delete color_R;
	//delete bestScale;
	//delete gx;
	//delete gy;
	//	delete original_image;

	
	long t4 = cv::getTickCount();
	std::cout << "Contours ....\n";
	size_track = contour_paths.size();
	for (size_t chain_nb = 0; chain_nb<size_track; chain_nb++)
	{
		contour_paths[chain_nb].smooth_pixel_colors();
		//vectorizes the left and right colors
		//vectorizes the blur
		contour_paths[chain_nb].vectorizeAttribs();
	}
	long t5 = cv::getTickCount();
	std::cout <<"Post processing time:"<<((t5-t1)/ cv::getTickFrequency())*1000 <<" "<<endl;
	/*
	//save XML
	for (size_t chain_nb = 0; chain_nb<size_track; chain_nb++)
	{
		curve_set.addCurve(contour_paths[chain_nb]);
	}
	//std::cout << "Saving data into XML....\n";
	//curve_set.saveXML2(xml_filename);
	curve_set.dealloc();
	*/
	//define the images for reconstruction
	//recompute the data at the indicated scale
	//

	CvScalar pC, pPos;

	IplImage* edges_reconstructed = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage* colorConstrains_reconstructed = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 3);


	//---------------------------------------------------------------------------------------------------------------------------------------------
	//POLYLINE with SMOOTHED COLORS
	cvZero(colorConstrains_reconstructed);
	cvZero(edges_reconstructed);


	//fill color sources image with -1
	pPos = cvScalar(-1, -1, -1, -1);
	//init
	for (int i = 0; i<height; i++) {
		for (int j = 0; j<width; j++) {
			cvSet2D(colorConstrains_reconstructed, i, j, pPos);
		}
	}


	QJsonObject jsonObj;
	QJsonArray curveArr;
	size_track = contour_paths.size();
	jsonObj["curve_count"] = static_cast<int>(size_track);
	
	for (size_t chain_nb = 0; chain_nb<size_track; chain_nb++)
	{
		contour_paths[chain_nb].draw(colorConstrains_reconstructed, NULL, NULL, NULL, edges_reconstructed, 2);
		contour_paths[chain_nb].toJson(curveArr);
	}
	jsonObj["curves"] = curveArr;
	QJsonDocument jsonDoc(jsonObj);
	QFile saveFile(QStringLiteral("").append(xml_filename));

	if (!saveFile.open(QIODevice::WriteOnly)) {
		qWarning("Couldn't open save file.");
	}
	saveFile.write(jsonDoc.toJson());

	width = imgf->width;
	height = imgf->height;
	IplImage* img_tmp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	CvScalar p;

	for (int i = 0; i<height; i++) {
		for (int j = 0; j<width; j++) {
			p = cvGet2D(colorConstrains_reconstructed, i, j);
			cvSet2D(img_tmp, i, j, p);
		}
	}

	//save_myimage(edges_reconstructed, "edgeMap", "WebUI/");
	save_myimage(colorConstrains_reconstructed, "colorSources", "WebUI/");

	//py::array_t<unsigned char>bEdge = cv_mat_uint8_3c_to_numpy(cv::cvarrToMat(edges_reconstructed));
	//py::array_t<unsigned char>csMap = cv_mat_uint8_3c_to_numpy(cv::cvarrToMat(colorConstrains_reconstructed));

	cv::Mat tmp = cv::cvarrToMat(img_tmp);
	cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
	py::list out;
	out.append<py::array_t<unsigned char>>(cv_mat_uint8_3c_to_numpy(cv::cvarrToMat(edges_reconstructed)));
	out.append<py::array_t<unsigned char>>(cv_mat_uint8_3c_to_numpy(tmp));
	//should release created images
	//should return contour sets
	return out;
}


PYBIND11_MODULE(pyVec, m) {
	m.doc() = "C++ wrapper for contour tracking"; // optional module docstring
	m.def("trackContour", &trackContour, "A function which track pixel chains and convert them to polylines/curves");
}