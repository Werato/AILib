using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_first
{
    internal class CSV_toImagge
    {
        public void Swow(string fileName, int frame = 0)
        {
            using (StreamWriter sw = File.CreateText("num.html"))
            {
                sw.WriteLine(ReadNumbersFromCSV(fileName, frame));
            }
            // Assuming you have a 2D array of numbers

            //Bitmap bitmap = new Bitmap(gridSize,
            //                           gridSize);

            //// Find the maximum number in the array
            //double maxNumber = FindMaxNumber(numbers);

            //// Loop through each number and set the corresponding pixel color
            //for (int x = 0; x < gridSize; x++)
            //{
            //    for (int y = 0; y < gridSize; y++)
            //    {
            //        double value = numbers[x, y];
            //        int pixelValue = (int)(value / maxNumber); // Scale the value to 0-255 range
            //        Color color = Color.FromArgb(pixelValue, pixelValue, pixelValue); // Grayscale color

            //        bitmap.SetPixel(x, y, color);
            //    }
            //}

            //// Save or display the bitmap
            //bitmap.Save("output_image.png");
            //Console.WriteLine("Image created successfully.");
        }
        public string ReadNumbersFromCSV(string filePath , int frame)
        {
            string table = "<table border=\"0\" width=\"1000\" style=\"height: 750px;border-collapse: collapse;\">";

            using (var reader = new StreamReader(filePath))
            {
                List<string> listA = new List<string>();
                List<string> listB = new List<string>();
                int inc = 0;
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',').Skip(1).ToArray();

                    if (values.Length % 2 != 0)
                    {
                        throw new Exception("alert it's not a square!!!!");
                    }

                    var squer = Convert.ToInt32(Math.Sqrt(values.Length));

                    if (frame == inc)
                    {

                        for (int i = 0; i < squer; i++)
                        {
                            var row = values.Skip(squer * i).Take(squer).ToArray();
                            table += CreateTableRow(row, squer);
                        }

                        break;
                    }
                    inc++; 
                }
            }
            table += "</table>";
            return table;
        }
        // Read the CSV and populate the numbers array
        public string CreateTableRow(string[]? brightness, int squer)
        {
            if (brightness == null)
            {
                return "";
            }
            string table = "<tr>";
            for (int x = 0; x < squer; x++)
            {
                table += $"<td " +
                    $"style='background-color: " +
                    $"rgb({brightness[x].ToString()}, {brightness[x].ToString()}, {brightness[x].ToString()})'>" +
                    $"</td>";
            }
            table += "</tr>";

            return table;
        }
        static double FindMaxNumber(double[,] numbers)
        {
            double max = double.MinValue;
            foreach (double number in numbers)
            {
                if (number > max)
                    max = number;
            }
            return max;
        }

        public Dictionary<int, List<int>> GetNumArray(string filePath)
        {
            Dictionary<int, List<int>> result = new Dictionary<int, List<int>>();

            using (var reader = new StreamReader(filePath))
            {
                List<string> listA = new List<string>();
                List<string> listB = new List<string>();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',').Skip(1).ToArray();

                    if (values.Length % 2 != 0)
                    {
                        throw new Exception("alert it's not a square!!!!");
                    }

                    var squer = Convert.ToInt32(Math.Sqrt(values.Length));

                    for (int i = 0; i < squer; i++)
                    {
                        result[i] = values.Skip(squer * i).Take(squer).Select(l => int.Parse(l)).ToList();
                    }
                    break;
                }
            }
            return result;
        }
    }
}