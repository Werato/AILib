using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Components
{
    public partial interface IAdder
    {
        public IEnumerable<IWeights> Weights { get; set; }

        public double Activation();//0..1
    }
}
