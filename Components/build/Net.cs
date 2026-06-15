using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection.PortableExecutable;
using System.Text;
using System.Threading.Tasks;

namespace Components
{
    public class Net 
    {
        
        public Net() { }
        public Net(List<int> hiddenLayers, int outputDim)
        {
            Net net = new Net();
            //28 * 28 = 784
            int inDem = hiddenLayers.Count();

            //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            Func<float> easy = () => (float)rand.NextDouble();
            //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            foreach (int outDem in hiddenLayers) 
            {
                Layers.Add(new Layer(inDem, outDem, easy));
                inDem = outDem;
            }

            Layers.Add(new Layer(inDem, outputDim, easy));
        }

        Random rand = new Random();
        public List<Layer> Layers = new();

        public void Run(string filePath)
        {
            List<float> input = new List<float>();
            List<float> output = new List<float>();
            using (var reader = new StreamReader(filePath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    input = line.Split(',').Skip(1)
                        .Select(l => float.Parse(l, CultureInfo.InvariantCulture.NumberFormat))
                        .ToList();
                }
            }

            foreach (var layer in Layers)
            {
                layer.CreateLayer(input, output);
            }


        }
    }
}
