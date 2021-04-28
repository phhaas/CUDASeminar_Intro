/*
    Simplified version of the code from "https://www.3dgep.com/introduction-opengl/".
*/

// includes, system
#include <math.h>
#include <iostream>
#include <ctime>

// OpenGL Graphics includes
#include "GL/glut.h"

float currentTime = 0;

int g_iWindowWidth = 512;
int g_iWindowHeight = 512;
int g_iGLUTWindowHandle = 0;
int g_iErrorCode = 0;

void InitGL(int argc, char* argv[]);
void DisplayGL();
void IdleGL();
void KeyboardGL(unsigned char c, int x, int y);
void MouseGL(int button, int state, int x, int y);
void MotionGL(int x, int y);
void ReshapeGL(int w, int h);

std::clock_t g_PreviousTicks;
std::clock_t g_CurrentTicks;

void RenderScene();

// We're exiting, cleanup the allocated resources.
void Cleanup(int exitCode, bool bExit = true);


int main(int argc, char* argv[])
{
    // Capture the previous time to calculate the delta time on the next frame
    g_PreviousTicks = std::clock();

    InitGL(argc, argv);
    //glutInit(&argc, argv);

    glutMainLoop();
    Cleanup(g_iErrorCode);
}

void Cleanup(int errorCode, bool bExit)
{
    if (g_iGLUTWindowHandle != 0)
    {
        glutDestroyWindow(g_iGLUTWindowHandle);
        g_iGLUTWindowHandle = 0;
    }

    if (bExit)
    {
        exit(errorCode);
    }
}


void InitGL(int argc, char* argv[])
{
    std::cout << "Initialise OpenGL..." << std::endl;

    glutInit(&argc, argv);
    int iScreenWidth = glutGet(GLUT_SCREEN_WIDTH);
    int iScreenHeight = glutGet(GLUT_SCREEN_HEIGHT);

    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);

    glutInitWindowPosition((iScreenWidth - g_iWindowWidth) / 2,
        (iScreenHeight - g_iWindowHeight) / 2);
    glutInitWindowSize(g_iWindowWidth, g_iWindowHeight);

    g_iGLUTWindowHandle = glutCreateWindow("OpenGL");

    // Register GLUT callbacks
    glutDisplayFunc(DisplayGL);
    glutIdleFunc(IdleGL);
    glutMouseFunc(MouseGL);
    glutMotionFunc(MotionGL);
    glutKeyboardFunc(KeyboardGL);
    glutReshapeFunc(ReshapeGL);

    // Setup initial GL State
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClearDepth(1.0f);

    glShadeModel(GL_SMOOTH);

    std::cout << "Initialise OpenGL: Success!" << std::endl;
}

void DisplayGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // Clear the color buffer, and the depth buffer.

    RenderScene();

    glutSwapBuffers();
    glutPostRedisplay();
}

void IdleGL()
{
    // Update our simulation
    g_CurrentTicks = std::clock();
    float deltaTicks = (g_CurrentTicks - g_PreviousTicks);
    g_PreviousTicks = g_CurrentTicks;

    float fDeltaTime = deltaTicks / (float)CLOCKS_PER_SEC;

    currentTime = g_CurrentTicks / (float)CLOCKS_PER_SEC;

    glutPostRedisplay();
}

void KeyboardGL(unsigned char c, int x, int y)
{

}

void MouseGL(int button, int state, int x, int y)
{

}

void MotionGL(int x, int y)
{

}

void ReshapeGL(int w, int h)
{
    std::cout << "ReshapGL( " << w << ", " << h << " );" << std::endl;

    if (h == 0)										// Prevent A Divide By Zero error
    {
        h = 1;										// Making Height Equal One
    }

    g_iWindowWidth = w;
    g_iWindowHeight = h;

    glViewport(0, 0, g_iWindowWidth, g_iWindowHeight);

    // Setup the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLdouble)g_iWindowWidth / (GLdouble)g_iWindowHeight, 0.1, 100.0);

    glutPostRedisplay();
}

void RenderScene()
{
    glMatrixMode(GL_MODELVIEW);                                           // Switch to modelview matrix mode
    glLoadIdentity();                                                       // Load the identity matrix

    glTranslatef(0.0f, 0.0f, -6.0f);                                     // Translate our view matrix back and a bit to the left.
    glColor3f(0.0f, 0.0f, 1.0f);                                          // Set Color to blue
    glBegin(GL_QUADS);
        glVertex2f(-2.0f, 2.0f);   // Top-Left
        glVertex2f(2.0f, 2.0f);   // Top-Right
        glVertex2f(2.0f, -2.0f);   // Bottom-Right
        glVertex2f(-2.0f, -2.0f);   // Bottom-Left
    glEnd();
}

