//
//  main.cpp
//  3Dtest
//
//  Created by David Huard on 2020-09-22.
//

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include <libraw.h>
#include <string.h>
#include <iostream>


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
/*
#define SQR(x) ((x) * (x))
void gamma_curve(unsigned short *curve, double *gamm, int imax)
{

    int mode = 2;
    int i;
    double g[6], bnd[2] = { 0, 0 }, r;

    g[0] = gamm[0];
    g[1] = gamm[1];
    g[2] = g[3] = g[4] = 0;
    bnd[g[1] >= 1] = 1;
    if (g[1] && (g[1] - 1) * (g[0] - 1) <= 0)
    {
        for (i = 0; i < 48; i++)
        {
            g[2] = (bnd[0] + bnd[1]) / 2;
            if (g[0])
                bnd[(pow(g[2] / g[1], -g[0]) - 1) / g[0] - 1 / g[2] > -1] = g[2];
            else
                bnd[g[2] / exp(1 - 1 / g[2]) < g[1]] = g[2];
        }
        g[3] = g[2] / g[1];
        if (g[0])
            g[4] = g[2] * (1 / g[0] - 1);
    }
    if (g[0])
        g[5] = 1 / (g[1] * SQR(g[3]) / 2 - g[4] * (1 - g[3]) +
        (1 - pow(g[3], 1 + g[0])) * (1 + g[4]) / (1 + g[0])) -
        1;
    else
        g[5] = 1 / (g[1] * SQR(g[3]) / 2 + 1 - g[2] - g[3] -
            g[2] * g[3] * (log(g[3]) - 1)) -
        1;

    memcpy(gamm, g, sizeof gamm);

    for (i = 0; i < 0x10000; i++)
    {
        curve[i] = 0xffff;
        if ((r = (double)i / imax) < 1)
            curve[i] =
            0x10000 *
            (mode ? (r < g[3] ? r * g[1]
                : (g[0] ? pow(r, g[0]) * (1 + g[4]) - g[4]
                    : log(r) * g[2] + 1))
                : (r < g[2] ? r / g[1]
                    : (g[0] ? pow((r + g[4]) / (1 + g[4]), 1 / g[0])
                        : exp((r - 1) / g[2]))));
    }
}

int FC(int row, int col, unsigned int filters)
{
    return (filters >> (((row << 1 & 14) | (col & 1)) << 1) & 3);
}
*/
int main(int argc, const char * argv[]) {
    
    
    string ImageDroitDossierPath = "/Users/davidhuard/Desktop/ImageD/*.png"; // Image Droite à traiter
    string ImageGaucheDossierPath = "/Users/davidhuard/Desktop/ImageG/*.png"; // Image Gauche à traiter
    string ImageDroitDossierPathHeic = "/Users/davidhuard/Desktop/ImageD/*.HEIC"; // Image Droite à traiter
    string ImageGaucheDossierPathHeic = "/Users/davidhuard/Desktop/ImageG/*.HEIC"; // Image Gauche à traiter
    const float resizeRatio = 0.10;//0.25 // Dimensionne l'image pour accélérer le processus.
    const float threshSearchRatio = 0.3f; // entre 0 et 1, 0 est très sévère pour le match. C'est le niveau de passage.
    const int thresholdBW = 50; // entre 0 et 255, threshold pour trouver la région d'intérêt (Crop)
    const string ImageGDMatchPath = "/Users/davidhuard/Desktop/ImageGDMatch/"; // Permet de voir l'image avec les matchs, l'image avec le plus de matchs déterminera les ajustements.
    const string ImageDthresPath = "/Users/davidhuard/Desktop/ImageDthres/"; // Permet de voir l'image transformé et l'encadrement avec le threshold
    const string ImageGDPath = "/Users/davidhuard/Desktop/ImageGD/"; // Permet de voir le résultat final.
  
    namedWindow( "Preview", WINDOW_AUTOSIZE );
    
    //Liste des images disponnibles dans les dossiers.
    
    cv::String pathGHeic(ImageGaucheDossierPathHeic);
    cv::String pathDHeic(ImageDroitDossierPathHeic);
    
    vector<cv::String> fnGHeic;
    vector<cv::String> fnDHeic;
    
    cout << fnGHeic.size() << endl;
    
    cv::glob(pathGHeic,fnGHeic,true);
    cv::glob(pathDHeic,fnDHeic,true);
    
    for(size_t k=0; k<fnDHeic.size(); ++k)
    {
    string outnameD = fnDHeic[k].substr(0, fnDHeic[k].length() - 4) + "png";
    string commandD = "/usr/local/Cellar/vips/8.10.2_1/bin/vips copy " + fnDHeic[k] + " " + outnameD;
    cout << "in: " << fnDHeic[k] << endl << "out: " << outnameD <<endl;
    system(commandD.c_str());
    }
    for(size_t k=0; k<fnGHeic.size(); ++k)
    {
    string outnameG = fnGHeic[k].substr(0, fnGHeic[k].length() - 4) + "png";
    string commandG = "/usr/local/Cellar/vips/8.10.2_1/bin/vips copy " + fnGHeic[k] + " " + outnameG;
    cout << "in: " << fnGHeic[k]<< endl << "out: " << outnameG <<endl;
    system(commandG.c_str());
    }
    
    cv::String pathG(ImageGaucheDossierPath);
    cv::String pathD(ImageDroitDossierPath);
    
    vector<cv::String> fnG;
    vector<cv::String> fnD;

    cv::glob(pathG,fnG,true);
    cv::glob(pathD,fnD,true);
    
    if(fnD.size()!=fnG.size())
    {
        std::cout << "Il doit avoir le meme nombre d'image dans les deux dossiers "  << std::endl;
        std::cout << "Dossier 1: " << fnD.size() << " images"  << std::endl;
        std::cout << "Dossier 2: " << fnG.size() << " images"  << std::endl;
        return 1;
    }
    else if(fnD.size()==0)
    {
        std::cout << "Les dossiers sont introuvable ou ils ne comportent pas de photo "  << std::endl;
    }
    
    int NbMaxMatch = 0;
    float yAjustement = 0;
    float xAjustement = 0;
    Rect RectCut = Rect(0,0,0,0);
    int maxHeight =0;
    
    for (size_t k=0; k<fnD.size(); ++k)
    {
        //Téléchargement des images par paires
        
        cout << "image: " << fnD[k] << endl;
        cout << "image: " << fnG[k] << endl;
        
        Mat imgD = imread(fnD[k], IMREAD_COLOR);
        Mat imgG = imread(fnG[k], IMREAD_COLOR);
        
        if(imgG.empty() )
        {
            std::cout << "Could not read the image: " << "G" << std::endl;
            return 1;
        }
        
        if( imgD.empty())
        {
            std::cout << "Could not read the image: " << "D" << std::endl;
            return 1;
        }
        
        //Dimensionnement de l'image
        
        resize(imgG, imgG, cv::Size(), resizeRatio, resizeRatio);
        resize(imgD, imgD, cv::Size(), resizeRatio, resizeRatio);
        
        //Match
        
        cv::Ptr<cv::SIFT> siftPtrG = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypointsG;
        siftPtrG->cv::SIFT::detect(imgG, keypointsG);
        
        cv::Ptr<cv::SIFT> siftPtrD = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypointsD;
        siftPtrD->cv::SIFT::detect(imgD, keypointsD);
        
        int minHessian = 400;//100
        Ptr<SURF> detector = SURF::create( minHessian );
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        detector->detectAndCompute( imgD, noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( imgG, noArray(), keypoints2, descriptors2 );
        
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < threshSearchRatio * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        
        Mat img_matches;
        drawMatches( imgD, keypoints1, imgG, keypoints2, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        string str = "Nombre de match: " + to_string(good_matches.size());
            cv::putText(img_matches, str,cv::Point(20,40), FONT_HERSHEY_PLAIN,2, CV_RGB(250,250,250),1);
        string filename = ImageGDMatchPath + "Image" + to_string(k) + ".png";
        cv::imwrite(filename.c_str(), img_matches);
        
        //"Crop"de l'image, aquisition des paramètres.
        
        Mat imgD_thres;
        Mat imgD_gray;
        cvtColor( imgD, imgD_gray, COLOR_BGR2GRAY );
        threshold( imgD_gray, imgD_thres, thresholdBW, 255, THRESH_BINARY );
        string filenameThres = ImageDthresPath + "Image" + to_string(k) + ".png";
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours( imgD_thres, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
        int t = 0;
        int maxPerimeter = 0;
        for(int y =0; y < contours.size(); y++)
        {
            int perimeter = arcLength(contours[y],false);
            if(perimeter>maxPerimeter)
            {
                t=y;
                maxPerimeter=perimeter;
            }
        }
        cv::Rect rect = cv::boundingRect(contours[t]);
        cout << "X: " << rect.x << " Y: " << rect.y << " W: " << rect.width << " H: " << rect.height << endl;
        rectangle(imgD_thres, rect, Scalar(255,255,255), 10,8,0 );
        cv::imwrite(filenameThres.c_str(), imgD_thres);
        
        cout << "perimeter : " << arcLength(contours[t],false) << endl;
        if(int(rect.height)>maxHeight)
        {
            maxHeight = int(rect.height);
            RectCut = Rect(rect.x, rect.y, rect.x+rect.width, rect.y+rect.height);
        }
        
        //Analyse du match.
        
        if(good_matches.size()>NbMaxMatch)
        {
            NbMaxMatch = int(good_matches.size());
            float yDiffMean = 0;
            float xDiffMean = 0;
            float j=0;
            for(size_t i = 0; i < good_matches.size(); i++)
            {
                float yDiff = keypoints1[good_matches[i].queryIdx].pt.y-keypoints2[good_matches[i].trainIdx].pt.y;
                if (yDiff<(100*resizeRatio) && yDiff>-(100*resizeRatio))
                {
                    j++;
                    yDiffMean += keypoints1[good_matches[i].queryIdx].pt.y-keypoints2[good_matches[i].trainIdx].pt.y;
                    xDiffMean += keypoints1[good_matches[i].queryIdx].pt.x-keypoints2[good_matches[i].trainIdx].pt.x;
                }
            }
            yAjustement = (yDiffMean/j)*(1/resizeRatio);
            xAjustement = (xDiffMean/j)*(1/resizeRatio);
        }

    }
    
    RectCut.y = int(RectCut.y)*(1/resizeRatio);
    RectCut.x = int(RectCut.x)*(1/resizeRatio);
    RectCut.height = int(RectCut.height)*(1/resizeRatio);
    RectCut.width = int(RectCut.width)*(1/resizeRatio);
    
    cout << "Coupure haute Y : " << RectCut.y << " - Coupure basse Y : " << RectCut.height << endl;
    cout << "Match : " << NbMaxMatch  << endl;
    cout << "Ajutement Y: " << yAjustement  << endl;
    cout << "Ajutement X: " << xAjustement  << endl;
 
    
    //Ajustement des images
    
    for (size_t k=0; k<fnD.size(); ++k)
    {
        Mat imgD = imread(fnD[k], IMREAD_COLOR);
        Mat imgG = imread(fnG[k], IMREAD_COLOR);
        
        Mat imgGD;
        
        Mat imgDcrop = imgD.clone();
        Mat imgGcrop = imgG.clone();
        /*
        if(xAjustement<0 && yAjustement<0)
        {
            Rect roiD = Rect(0,rectCut.x, imgD.cols+xAjustement, imgD.rows+yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgDcrop = imgD(roiD).clone() ;
            
            Rect roiG = Rect(-xAjustement,
         -yAjustement+rectCut.x,
         imgG.cols+xAjustement,
         imgG.rows+yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgGcrop = imgG(roiG).clone() ;
        }*/
        
        if(xAjustement<0 && yAjustement<0)
        {
            Rect roiD = Rect(RectCut.x,RectCut.y, imgD.cols+xAjustement-RectCut.x-(imgD.cols-RectCut.width), imgD.rows+yAjustement-RectCut.y-(imgD.rows-RectCut.height));
            imgDcrop = imgD(roiD).clone() ;
            
            Rect roiG = Rect(-xAjustement+RectCut.x,-yAjustement+RectCut.y, imgG.cols+xAjustement-RectCut.x-(imgG.cols-RectCut.width), imgG.rows+yAjustement-RectCut.y-(imgG.rows-RectCut.height));
            imgGcrop = imgG(roiG).clone() ;
        }
        
        /*
        else if(xAjustement>0 && yAjustement>0)
        {
            Rect roiG = Rect(0,rectCut.x, imgG.cols-xAjustement, imgG.rows-yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgGcrop = imgG(roiG).clone() ;
            
            Rect roiD = Rect(xAjustement,yAjustement+rectCut.x, imgD.cols-xAjustement, imgD.rows-yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgDcrop = imgD(roiD).clone() ;
        }*/
        else if(xAjustement>0 && yAjustement>0)
        {
            Rect roiG = Rect(RectCut.x,RectCut.y, imgG.cols-xAjustement-RectCut.x-(imgG.cols-RectCut.width), imgG.rows-yAjustement-RectCut.y-(imgG.rows-RectCut.height));
            imgGcrop = imgG(roiG).clone() ;
            
            Rect roiD = Rect(xAjustement+RectCut.x,yAjustement+RectCut.y, imgD.cols-xAjustement-RectCut.x-(imgG.cols-RectCut.width), imgD.rows-yAjustement-RectCut.y-(imgG.rows-RectCut.height));
            imgDcrop = imgD(roiD).clone() ;
        }
         /*else if(xAjustement>0 && yAjustement<0)
        {
            Rect roiG = Rect(0,rectCut.x, imgG.cols-xAjustement, imgD.rows+yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgGcrop = imgG(roiG).clone() ;
            
            Rect roiD = Rect(xAjustement,-yAjustement+rectCut.x, imgD.cols-xAjustement, imgG.rows+yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgDcrop = imgD(roiD).clone() ;
        }*/
         else if(xAjustement>0 && yAjustement<0)
        {
            
            Rect roiG = Rect(RectCut.x,RectCut.y, imgG.cols-xAjustement-RectCut.x-(imgG.cols-RectCut.width), imgG.rows+yAjustement-RectCut.y-(imgG.rows-RectCut.height));
            imgGcrop = imgG(roiG).clone() ;
            
            Rect roiD = Rect(xAjustement+RectCut.x,-yAjustement+RectCut.y, imgD.cols-xAjustement-RectCut.x-(imgG.cols-RectCut.width), imgD.rows+yAjustement-RectCut.y-(imgG.rows-RectCut.height));
            imgDcrop = imgD(roiD).clone() ;
        }/*
         else if(xAjustement<0 && yAjustement>0)
         {
             Rect roiG = Rect(0,rectCut.x, imgG.cols+xAjustement, imgG.rows-yAjustement-rectCut.x-(imgG.rows-rectCut.y));
             imgGcrop = imgG(roiG).clone() ;
             
             Rect roiD = Rect(-xAjustement,yAjustement+rectCut.x, imgD.cols+xAjustement, imgD.rows-yAjustement-rectCut.x-(imgG.rows-rectCut.y));
             imgDcrop = imgD(roiD).clone() ;
         }*/
        else if(xAjustement<0 && yAjustement>0)
        {
            Rect roiD = Rect(RectCut.x,RectCut.y, imgD.cols+xAjustement-RectCut.x-(imgD.cols-RectCut.width), imgD.rows-yAjustement-RectCut.y-(imgD.rows-RectCut.height));
            imgDcrop = imgD(roiD).clone() ;
            
            Rect roiG = Rect(-xAjustement+RectCut.x,yAjustement+RectCut.y, imgG.cols+xAjustement-RectCut.x-(imgG.cols-RectCut.width), imgG.rows-yAjustement-RectCut.y-(imgG.rows-RectCut.height));
            imgGcrop = imgG(roiG).clone() ;
            
        }
        
        //Ajutement Brightness
        
        Mat imgDHSV;
        cvtColor(imgDcrop, imgDHSV, cv::COLOR_RGB2HSV_FULL);
        Mat imgGHSV;
        cvtColor(imgGcrop, imgGHSV, cv::COLOR_RGB2HSV_FULL);
        
        Mat imgOut = imgDcrop.clone();
        
        
        cv::Scalar tempValD = cv::mean( imgDHSV );
        float myMAtMeanD = tempValD.val[2];
        cv::Scalar tempValG = cv::mean( imgGHSV );
        float myMAtMeanG = tempValG.val[2];
    
        int diff = myMAtMeanG-myMAtMeanD;
        cout << "Diff: " << diff <<endl;
  
        for( int y = 0; y < imgDHSV.rows; y++ ) {
                for( int x = 0; x < imgDHSV.cols; x++ ) {
                    
                    imgDcrop.at<Vec3b>(y,x)[0] = saturate_cast<uchar>(imgDcrop.at<Vec3b>(y,x)[0] + diff );
                    imgDcrop.at<Vec3b>(y,x)[1] = saturate_cast<uchar>(imgDcrop.at<Vec3b>(y,x)[1] + diff );
                    imgDcrop.at<Vec3b>(y,x)[2] = saturate_cast<uchar>(imgDcrop.at<Vec3b>(y,x)[2] + diff );
                }
            }
        

        //SBS image
        
        hconcat(imgGcrop,imgDcrop,imgGD);

        string filename = ImageGDPath + "Image" + to_string(k) + ".png";
        cv::imwrite(filename.c_str(), imgGD);
  
    }
    
    /*
    LibRaw RawProcessor;
    RawProcessor.imgdata.params.output_bps = 16;
    
    libraw_processed_image_t *proc_img = NULL;
    
    char av[] = "/Users/davidhuard/Desktop/VanHerick_Lemire_01.cr3";

    
    // Read RAW image
     RawProcessor.open_file(av);
     RawProcessor.unpack();
    // white balance + color interpolation + colorspace conversion
    int ret = RawProcessor.dcraw_process();

    // gamma correction + create 3 component Bitmap
    proc_img = RawProcessor.dcraw_make_mem_image(&ret);
    
    libraw_processed_image_t* output = RawProcessor.dcraw_make_mem_image(&ret);
    
    Mat mat16uc3_rgb(RawProcessor.imgdata.sizes.raw_height, RawProcessor.imgdata.sizes.raw_width, CV_16UC1, output->data);
    //imwrite("/Users/davidhuard/Desktop/out_manual.tiff", proc_img);
    imshow("Preview",mat16uc3_rgb);
     RawProcessor.recycle();
     */
    
    
    //imshow("Preview",RGBImage);
    
    
    
   // waitKey(0);
    
  
  
    return 0;
}
