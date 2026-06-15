using Components;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AILib
{
    public class Read
    {
        public string Document;
        public Read(string fileName) 
        {
            Document = fileName;
            //using (var reader = new StreamReader(filePath)) { }
        }



    }
}
