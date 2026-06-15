using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Components
{
    public class Weights : IWeights
    {
        public int Signal { get; set; }
        public float Weight { get; set; }

        public float Outputter => Signal * Weight;
    }
}
