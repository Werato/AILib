using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Components
{
    public partial interface ILayer
    {
        public float Forvard(float[] input);
        public float Backward(float[] outputGradient, float learningRate);
    }
}
