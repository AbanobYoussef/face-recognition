using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using Emgu.CV;
using Emgu.CV.Structure;

namespace WindowsFormsApplication1
{
    public partial class Form1 : Form
    {
        // declare variables to use in this project
        MCvFont font = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_TRIPLEX, 0.6d, .6d);
        HaarCascade facedetected;
        Image<Bgr, byte> Frame;
        Capture camera;
        Image<Gray, byte> result;
        Image<Gray, byte> TrainedFace = null;
        Image<Gray, byte> grayFace = null;
        List<Image<Gray, byte>> trainningimages = new List<Image<Gray, byte>>();
        List<string> lables=new List<string>();
        List<string> users = new List<string>();
        int count, Numlable, t;
        string name, names = null;
        public Form1()
        {
            InitializeComponent();
            camera = new Capture();
            Application.Idle += new EventHandler(Frameprocedure);
            // haar for face deteaction
            facedetected = new HaarCascade("haarcascade_frontalface_default.xml");
             try
            {
                string lableinfo = File.ReadAllText(Application.StartupPath+"/Faces/Faces.txt");
                string[] Lables = lableinfo.Split(',');
                 // the first lable will contant the number of the faces
                Numlable = Convert.ToInt16(Lables[0]);
                count = Numlable;
                string Faceload;
                 for(int i =1 ;i <Numlable+1;i++)
                 {
                     Faceload = "faces" + i + ".bmp";
                     trainningimages.Add(new Image<Gray, byte>(Application.StartupPath + "/Faces/"+Faceload));
                     lables.Add(Lables[i]);
                 }

            }
            catch(Exception ex)
             {
                 //MessageBox.Show("Nothing in the database");
             }
        }

        private void button1_Click(object sender, EventArgs e)
        {
           
        }
        private void Frameprocedure (object sender , EventArgs e)
        {
            users.Add("");
            Frame= camera.QueryFrame().Resize(imageBox1.Width,imageBox1.Height,Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            grayFace=Frame.Convert<Gray,Byte>();
            MCvAvgComp[][]facedetectedNow = grayFace.DetectHaarCascade(facedetected,1.2,10,Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,new Size(20,20));
            
            foreach(MCvAvgComp f in facedetectedNow[0])
            {
                result = Frame.Copy(f.rect).Convert<Gray,byte>().Resize(100,100,Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                Frame.Draw(f.rect, new Bgr(Color.Green), 3);
                if(trainningimages.ToArray().Length !=0)
                {
                    MCvTermCriteria termcriterias = new MCvTermCriteria(count, 0.001);
                    EigenObjectRecognizer recognizer = new EigenObjectRecognizer(trainningimages.ToArray(), lables.ToArray(), 1500, ref termcriterias);
                    name = recognizer.Recognize(result);
                    Frame.Draw(name,ref font,new Point(f.rect.X - 2, f.rect.Y - 2), new Bgr(Color.Red));

                }
                
                users.Add("");
           }
            imageBox1.Image = Frame;
            names = "";
            users.Clear();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            count += 1;
            grayFace = camera.QueryGrayFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            MCvAvgComp[][] Detectedfaces = grayFace.DetectHaarCascade(facedetected,1.2,10,Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,new Size(20,20));
            foreach (MCvAvgComp f in Detectedfaces[0])
            {
                TrainedFace = Frame.Copy(f.rect).Convert<Gray, byte>();
                break;
            }
            TrainedFace = result.Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            trainningimages.Add(TrainedFace);
            File.WriteAllText(Application.StartupPath + "/Faces/Faces.txt",trainningimages.ToArray().Length.ToString()+",");
            for (int i =1;i<trainningimages.ToArray().Length+1;i++)
            {
                trainningimages.ToArray()[i - 1].Save(Application.StartupPath + "/Faces/faces" + i + ".bmp");
                File.AppendAllText(Application.StartupPath + "/Faces/Faces.txt", lables.ToArray()[i-1] + ",");
            }
        }
    }
}
