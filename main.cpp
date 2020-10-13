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
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;



int main(int argc, const char * argv[]) {
    
    string ImageDroitDossierPath = "/Users/davidhuard/Desktop/ImageG/*.png"; // Image Droite à traiter
    string ImageGaucheDossierPath = "/Users/davidhuard/Desktop/ImageD/*.png"; // Image Gauche à traiter
    const float resizeRatio = 0.25; // Dimensionne l'image pour accélérer le processus.
    const float threshSearchRatio = 0.4f; // entre 0 et 1, 0 est très sévère pour le match. C'est le niveau de passage.
    const int thresholdBW = 50; // entre 0 et 255, threshold pour trouver la région d'intérêt (Crop)
    const string ImageGDMatchPath = "/Users/davidhuard/Desktop/ImageGDMatch/"; // Permet de voir l'image avec les matchs, l'image avec le plus de matchs déterminera les ajustements.
    const string ImageDthresPath = "/Users/davidhuard/Desktop/ImageDthres/"; // Permet de voir l'image transformé et l'encadrement avec le threshold
    const string ImageGDPath = "/Users/davidhuard/Desktop/ImageGD/"; // Permet de voir le résultat final.
  
    namedWindow( "G", WINDOW_AUTOSIZE );
    namedWindow( "GD", WINDOW_AUTOSIZE );
    
    //Liste des images disponnibles dans les dossiers.
    
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
    Point rectCut = Point(0,0);
    
    
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
        
        int minHessian = 100;//400
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
        
        //"Crop"de l'image, aquisition des paramètre.
        
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
        rectCut = rectCut+ Point(rect.y, rect.y+rect.height);
        
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
    rectCut.x = int(rectCut.x/fnD.size())*(1/resizeRatio);
    rectCut.y = int(rectCut.y/fnD.size())*(1/resizeRatio);
    cout << "Coupure haute Y : " << rectCut.x << " - Coupure basse Y : " << rectCut.y << endl;
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
        
        if(xAjustement<0 && yAjustement<0)
        {
            Rect roiD = Rect(0,rectCut.x, imgD.cols+xAjustement, imgD.rows+yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgDcrop = imgD(roiD).clone() ;
            
            Rect roiG = Rect(-xAjustement,-yAjustement+rectCut.x, imgG.cols+xAjustement, imgG.rows+yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgGcrop = imgG(roiG).clone() ;
        }
        else if(xAjustement>0 && yAjustement>0)
        {
            Rect roiG = Rect(0,rectCut.x, imgG.cols-xAjustement, imgG.rows-yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgGcrop = imgG(roiG).clone() ;
            
            Rect roiD = Rect(xAjustement,yAjustement+rectCut.x, imgD.cols-xAjustement, imgD.rows-yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgDcrop = imgD(roiD).clone() ;
        }
         else if(xAjustement>0 && yAjustement<0)
        {
            Rect roiG = Rect(0,rectCut.x, imgG.cols-xAjustement, imgD.rows+yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgGcrop = imgG(roiG).clone() ;
            
            Rect roiD = Rect(xAjustement,-yAjustement+rectCut.x, imgD.cols-xAjustement, imgG.rows+yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgDcrop = imgD(roiD).clone() ;
        }
        else if(xAjustement<0 && yAjustement>0)
        {
            Rect roiG = Rect(0,rectCut.x, imgG.cols+xAjustement, imgG.rows-yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgGcrop = imgG(roiG).clone() ;
            
            Rect roiD = Rect(-xAjustement,yAjustement+rectCut.x, imgD.cols+xAjustement, imgD.rows-yAjustement-rectCut.x-(imgG.rows-rectCut.y));
            imgDcrop = imgD(roiD).clone() ;
        }
        
        hconcat(imgDcrop,imgGcrop,imgGD);

        string filename = ImageGDPath + "Image" + to_string(k) + ".png";
        cv::imwrite(filename.c_str(), imgGD);
    }
  
    return 0;
}
