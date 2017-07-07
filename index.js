var ndarray = require("ndarray");
var convnetjs = require("convnetjs");
var Jimp = require("jimp");

let dim = 28;
let depth = 1;
let size = dim * dim;

let input = {type:'input', out_sx:dim, out_sy:dim, out_depth: 1};
let conv = {type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'};
let pool = {type:'pool', sx:2, stride:2};
let output = {type:'softmax', num_classes:10};

let layers = [
    input,
    conv,
    pool,
    conv,
    pool,
    output
];

let net = new convnetjs.Net();
net.makeLayers(layers); 

let trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001,
                                    batch_size: 20});
                                    
let labels = ['0','1','2','3','4','5','6','7','8','9'];

for(let label of labels) {
    const training = require('./digits/' + label + '.json');

    const data = training.data;

    let curPixel = 1;
    let curImage = 1;
    let points = [];

    for(const d in data){
        let value = parseFloat(data[d]);
        points.push(value);
        if(curPixel++ >= size) {
            let image = new convnetjs.Vol(dim, dim, depth);
            image.w = points;
            trainer.train(image, label);
            console.log('Training Image ' + (curImage++) + ' for Label ' + label);
            
            if(curImage%1000 == 0){
                let png = new Jimp(dim, dim, function (err, png) {
                  if (err) throw err;
                  
                  let arr = ndarray(new Float64Array(points), [dim,dim]);
                  let nx = arr.shape[0], 
                      ny = arr.shape[1];

                  for(var i=1; i<nx-1; ++i) {
                    for(var j=1; j<ny-1; ++j) {
                        let value = arr.get(i, j);
                        let nrmValue = parseInt(Math.round(value * 255));
                        let hex = Jimp.rgbaToInt(nrmValue, 0, 0, 1);
                        png.setPixelColor(hex, i, j);
                    }
                  }
                  let fileName = 'out/test_' + curImage + '_' + label + '.bmp';

                  png.write(fileName, (err) => {
                    if (err) throw err;
                    else console.log('Wrote ' + fileName);
                  });
                });
            }
            
            curPixel = 1;
            points = [];
        }
    }

}     

for(let label of labels) {
    const testing = require('./digits-test/' + label + '.json');

    const data = testing.data;

    let curPixel = 1;
    let curImage = 1;
    let flatimage = [];

    for(let i in data){
        let value = parseFloat(data[i]);
        flatimage.push(value);

        if(curPixel++ >= size) {
            let image = new convnetjs.Vol(dim, dim, depth);
            image.w = flatimage;
            let results = net.forward(image);
            console.log('Testing Image ' + (curImage++) + ' for label ' + label, results);
            curPixel = 1;
            flatimage = [];
        }
    }

}              

















