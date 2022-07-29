static int n = 0;

class Direction {
  float px, py;
  float dx, dy;
  
  Direction(float _px, float _py, float _dx, float _dy)
  {
    px = _px;
    py = _py;
    dx = _dx;
    dy = _dy;
  }
}

class Line
{
  public float x1,y1,x2,y2;
  public float w;
    
  Line(float _x1, float _y1, float _x2, float _y2, float _w){
    x1= _x1;
    y1 = _y1;
    x2 = _x2;
    y2 = _y2;
    w = _w;
  }
  
  Direction pointOnLine(){
    float t = random(0.0,1.0);
    float dx = x2-x1;
    float dy = y2-y1;
    return new Direction(t*dx, t*dy, -dy, dx);
  }
}

class Bezier
{
  public float x1,y1,x2,y2, x3,y3,x4,y4;
  public float w;
  
  Bezier(float _x1, float _y1, float _x2, float _y2, float _x3, float _y3, float _x4, float _y4, float _w){
    x1 = _x1;
    y1 = _y1;
    x2 = _x2;
    y2 = _y2;
    x3 = _x3;
    y3 = _y3;
    x4 = _x4;
    y4 = _y4;
    w = _w;
  }
}

int linecount;
Line[] lines;
int beziercount;
Bezier[] beziers;

void setup(){
  size(400,400);
  background(0);
  keyPressed();
}

void draw() {
   clear();
   stroke(255);
   noFill();
  
  for (int i = 0; i < linecount; i++){
    strokeWeight(lines[i].w);
    line(lines[i].x1,lines[i].y1,lines[i].x2,lines[i].y2);
  }
  
  for (int i = 0; i < beziercount; i++){
    strokeWeight(beziers[i].w);
    bezier(beziers[i].x1, beziers[i].y1, beziers[i].x2, beziers[i].y2, beziers[i].x3, beziers[i].y3, beziers[i].x4, beziers[i].y4);
  }
}

void keyPressed() {
  if(key == 's'){
    save("generated"+n+"_seg.png");
    n++;
  } else {
    
    linecount = int(random(2,6));
    beziercount = int(random(0,4));
    lines = new Line[linecount];
    beziers = new Bezier[beziercount];
    
    for (int i = 0; i < linecount; i++){
      float chance = random(0,1);
      if(i==0){
        print("r");
        float x1 = random(-10,410);
        float y1 = random(-10,410);
        float x2 = random(-10,410);
        float y2 = random(-10,410);
        float w = random(4,11);
        lines[i] = new Line(x1,y1,x2,y2,w);
        if(chance < 0.4 & i < linecount-1){
          i++;
          lines[i] = new Line(x1+2*w, y1, x2+2*w, y2, w);
        }
      } else if (i > 0){
        if(chance > 0.7){
          print("a");
          // attach to previous line orthogonally
          Direction d = lines[i-1].pointOnLine();
          float flip = random(2);
          float f = flip < 1.0 ? 1.0 : -1.0;
          float l = random(20,400);
          float x1 = d.px;
          float y1 = d.py;
          float x2 = d.px + f*d.dx*l;
          float y2 = d.py + f*d.dy*l;
          float w = lines[i-1].w * random(0.5,1);
          lines[i] = new Line(x1,y1,x2,y2,w);
        } else {
          // screen crossing
          float dir = random(0,1);
          float x = random(-10,410);
          float y = random(-10,410);
          if(dir > 0.5){ // horz
            print("h");
            float x1 = -10;
            float y1 = x;
            float x2 = 410;
            float y2 = y;
            float w = random(6,10);
            lines[i] = new Line(x1,y1,x2,y2,w);
          } else { // vert
            print("v");
            float x1 = x;
            float y1 = -10;
            float x2 = y;
            float y2 = 410;
            float w = random(6,10);
            lines[i] = new Line(x1,y1,x2,y2,w);
          }
        }
      }
    }
    print("\n");
    
    for (int i = 0; i < beziercount; i++)
    {
      float x1 = random(0,400);
      float y1 = random(0,400);
      float x2 = random(0,400);
      float y2 = random(0,400);
      float x3 = random(0,400);
      float y3 = random(0,400); 
      float x4 = random(0,400);
      float y4 = random(0,400);
      float w = random(4,8);
      beziers[i] = new Bezier(x1,y1,x2,y2,x3,y3,x4,y4,w);
    }
  }
}
