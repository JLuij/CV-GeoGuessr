let nodes = [];

var cell_index = 0

function setup(){
  let canvas = createCanvas(min(windowWidth, windowHeight/0.8), min(0.8 * windowWidth, windowHeight));  //zet de schermgrote

  // Move the canvas so it’s inside our <div id="sketch-holder">.
  canvas.parent('sketch-holder');

  set_nodes()

}

function windowResized() {
  resizeCanvas(min(windowWidth, windowHeight/0.8), min(0.8 * windowWidth, windowHeight));
}

function draw(){
  background(255);

  ellipse(mouseX, mouseY, 33, 33);

  let closest = 1000
  for(let n in nodes){
    let node_x = map_x(nodes[n].x)
    let node_y = map_y(nodes[n].y)
    let distance =  (node_x - mouseX) * (node_x - mouseX) +
      (node_y - mouseY) * (node_y - mouseY)
    if(distance < closest){
      closest = distance
      cell_index = n
    }
  }


  for(let n in nodes){
    nodes[n].show(cell_index);
  }

  drawLondon()

}
