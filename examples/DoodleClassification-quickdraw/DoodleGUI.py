import pygame
import numpy as np
import sys
import os

from neuralnet import NeuralNetwork, Matrix

# Layout Configuration
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 650
CANVAS_SIZE = 560  # 28 * 20 (High resolution drawing area)
INPUT_SIZE = 28    # Neural Network Input Size
FPS = 120

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BG_COLOR = (30, 30, 30)
PANEL_COLOR = (45, 45, 45)
ACCENT_COLOR = (0, 200, 100) # Green
ERROR_COLOR = (220, 50, 50)  # Red
TEXT_MAIN = (240, 240, 240)
TEXT_DIM = (160, 160, 160)

class Button:
    def __init__(self, x, y, w, h, text, color, action=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.base_color = color
        self.action = action
        self.hover = False

    def draw(self, screen, font):
        # Calculate display color (lighter when hovered)
        color = self.base_color
        if self.hover:
            color = tuple(min(c + 30, 255) for c in self.base_color)
        
        # Draw Shadow
        pygame.draw.rect(screen, (20, 20, 20), (self.rect.x+2, self.rect.y+4, self.rect.width, self.rect.height), border_radius=8)
        # Draw Main Button
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        
        # Text
        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.hover and self.action:
                self.action()

class DoodleApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("NeuralNet Doodle Classifier (High-Res)")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_small = pygame.font.SysFont("Verdana", 14)
        self.font_btn = pygame.font.SysFont("Verdana", 16, bold=True)
        self.font_large = pygame.font.SysFont("Verdana", 24, bold=True)
        self.font_giant = pygame.font.SysFont("Verdana", 32, bold=True)
        
        # Drawing Canvas (High Res)
        self.canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        self.canvas.fill(BLACK)
        
        # Layout positions
        self.canvas_x = 20
        self.canvas_y = (WINDOW_HEIGHT - CANVAS_SIZE) // 2
        
        self.panel_x = self.canvas_x + CANVAS_SIZE + 20
        self.panel_w = WINDOW_WIDTH - self.panel_x - 20
        
        # Brush settings
        self.brush_size = 18
        self.last_pos = None
        self.drawing = False
        
        # Load Model
        self.model_status = "Loading..."
        try:
            print("Loading best model...")
            self.nn = NeuralNetwork.fromFile("DoodleClassificationBest.json")
            print("Model loaded.")
            self.model_status = "Ready"
        except Exception as e:
            print(f"Error loading model: {e}")
            self.nn = None
            self.model_status = "Error loading model"

        # Predictions
        self.top_pred = "Start Drawing"
        self.top_conf = 0.0
        self.predictions = [] # List of dicts
        
        # Buttons
        btn_y = WINDOW_HEIGHT - 80
        btn_w = (self.panel_w - 20) // 2
        self.buttons = [
            Button(self.panel_x, btn_y, btn_w, 50, "CLEAR", ERROR_COLOR, self.clear_canvas),
            Button(self.panel_x + btn_w + 20, btn_y, btn_w, 50, "PREDICT", ACCENT_COLOR, self.predict)
        ]

    def clear_canvas(self):
        self.canvas.fill(BLACK)
        self.top_pred = "Draw!"
        self.confidence = 0.0
        self.predictions = []

    def predict(self):
        if self.nn is None: return

        # 1. Downscale Logic
        # Resize high-res canvas to 28x28
        small_surface = pygame.transform.smoothscale(self.canvas, (INPUT_SIZE, INPUT_SIZE))
        
        # 2. Extract Data
        # Get RGB array (28, 28, 3)
        pixels3d = pygame.surfarray.array3d(small_surface)
        
        # Take Red channel (since we draw white on black, R=G=B)
        # Transpose: Pygame is (x, y), we need (row, col) i.e. (y, x)
        grayscale = pixels3d[:, :, 0].T 
        
        # 3. Normalize (0-255 -> 0.0-1.0)
        input_data = (grayscale / 255.0).flatten().tolist()
        
        # 4. Predict
        try:
            results = self.nn.classify(input_data)
            self.predictions = results # store all results
            
            top = results[0]
            self.top_pred = top['class']
            self.top_conf = top['confidence']
            print(f"Prediction: {self.top_pred} ({self.top_conf:.2%})")
            
        except Exception as e:
            print(f"Pred Error: {e}")
            self.top_pred = "Error"

    def draw_brush(self, start, end):
        # Draw a smooth line for the brush
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = max(abs(dx), abs(dy))
        
        for i in range(distance):
            x = int(start[0] + float(i) / distance * dx)
            y = int(start[1] + float(i) / distance * dy)
            pygame.draw.circle(self.canvas, WHITE, (x, y), self.brush_size)
        pygame.draw.circle(self.canvas, WHITE, end, self.brush_size)

    def handle_input(self):
        mouse_pos = pygame.mouse.get_pos()
        x, y = mouse_pos
        
        rel_x = x - self.canvas_x
        rel_y = y - self.canvas_y
        
        mouse_in_canvas = (0 <= rel_x < CANVAS_SIZE and 0 <= rel_y < CANVAS_SIZE)
        
        # Check mouse state
        click = pygame.mouse.get_pressed()[0]
        
        if click and mouse_in_canvas:
            if not self.drawing:
                # Just started
                self.drawing = True
                self.last_pos = (rel_x, rel_y)
                pygame.draw.circle(self.canvas, WHITE, (rel_x, rel_y), self.brush_size)
            else:
                # Continue drawing
                self.draw_brush(self.last_pos, (rel_x, rel_y))
                self.last_pos = (rel_x, rel_y)
            
            # Auto-predict if user wants? (Optional, maybe laggy on python-only NN)
            # self.predict() 
        else:
            self.drawing = False
            self.last_pos = None

    def draw_ui(self):
        # Background
        self.screen.fill(BG_COLOR)
        
        # Draw Canvas Frame
        pygame.draw.rect(self.screen, (60, 60, 60), 
                         (self.canvas_x-4, self.canvas_y-4, CANVAS_SIZE+8, CANVAS_SIZE+8), border_radius=4)
        self.screen.blit(self.canvas, (self.canvas_x, self.canvas_y))
        
        # Right Panel
        
        # Title
        title_surf = self.font_large.render("Doodle Classifier", True, TEXT_MAIN)
        self.screen.blit(title_surf, (self.panel_x, 30))
        
        status_color = ACCENT_COLOR if self.nn else ERROR_COLOR
        status_surf = self.font_small.render(f"System: {self.model_status}", True, status_color)
        self.screen.blit(status_surf, (self.panel_x, 70))
        
        # Separator
        pygame.draw.line(self.screen, PANEL_COLOR, (self.panel_x, 100), (WINDOW_WIDTH-20, 100), 2)
        
        # Prediction Display
        header_surf = self.font_small.render("PREDICTION", True, TEXT_DIM)
        self.screen.blit(header_surf, (self.panel_x, 120))
        
        pred_surf = self.font_giant.render(self.top_pred, True, WHITE)
        self.screen.blit(pred_surf, (self.panel_x, 150))
        
        if self.top_conf > 0:
            conf_str = f"Confidence: {self.top_conf*100:.1f}%"
            conf_surf = self.font_large.render(conf_str, True, ACCENT_COLOR)
            self.screen.blit(conf_surf, (self.panel_x, 210))
            
            # Draw Horizontal Bar
            bar_w = self.panel_w
            bar_h = 10
            fill_w = int(bar_w * self.top_conf)
            pygame.draw.rect(self.screen, PANEL_COLOR, (self.panel_x, 250, bar_w, bar_h), border_radius=5)
            pygame.draw.rect(self.screen, ACCENT_COLOR, (self.panel_x, 250, fill_w, bar_h), border_radius=5)

        # Other Guesses
        if len(self.predictions) > 1:
            y_off = 300
            pygame.draw.line(self.screen, PANEL_COLOR, (self.panel_x, 280), (WINDOW_WIDTH-20, 280), 1)
            other_head = self.font_small.render("Other Guesses:", True, TEXT_DIM)
            self.screen.blit(other_head, (self.panel_x, 290))
            
            for i in range(1, min(4, len(self.predictions))):
                p = self.predictions[i]
                text = f"{i}. {p['class']} ({p['confidence']*100:.1f}%)"
                surf = self.font_small.render(text, True, TEXT_DIM)
                self.screen.blit(surf, (self.panel_x, 320 + (i-1)*25))


        # Buttons
        for btn in self.buttons:
            btn.draw(self.screen, self.font_btn)

    def run(self):
        running = True
        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                for btn in self.buttons:
                    btn.handle_event(event)
                    
            # Update
            self.handle_input()
            
            # Draw
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = DoodleApp()
    app.run()
