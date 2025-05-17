#[macro_use]
extern crate glium;
extern crate winit;

use nalgebra::{Matrix4, Perspective3, Point3, Vector3}; // Add nalgebra for matrix calculations
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone)]
struct Bird {
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    acceleration: Vector3<f32>,
}

impl Bird {
    fn new<R: Rng>(rng: &mut R) -> Self {
        Bird {
            position: Vector3::new(rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)),
            velocity: Vector3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)),
            acceleration: Vector3::zeros(),
        }
    }
}

const NUM_BIRDS: usize = 100;

const DIMENSIONS: f32 = 10.0;
const SPACE_MIN: f32 = -DIMENSIONS;
const SPACE_MAX: f32 = DIMENSIONS;

fn wraparound(mut v: Vector3<f32>) -> Vector3<f32> {
    for i in 0..3 {
        if v[i] < SPACE_MIN {
            v[i] = SPACE_MAX - (SPACE_MIN - v[i]) % (SPACE_MAX - SPACE_MIN);
        } else if v[i] > SPACE_MAX {
            v[i] = SPACE_MIN + (v[i] - SPACE_MAX) % (SPACE_MAX - SPACE_MIN);
        }
    }
    v
}

fn main() {
    #[allow(unused_imports)]
    use glium::{glutin, Surface};

    let event_loop = glium::winit::event_loop::EventLoop::builder()
        .build()
        .expect("event loop building");
    let (window, display) = glium::backend::glutin::SimpleWindowBuilder::new()
        .with_title("Triangles")
        .build(&event_loop);

    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
    }

    implement_vertex!(Vertex, position);

    let vertex1 = Vertex { position: [-0.05, -0.0288] };
    let vertex2 = Vertex { position: [ 0.00,  0.0577] };
    let vertex3 = Vertex { position: [ 0.05, -0.0288] };
    let shape = vec![vertex1, vertex2, vertex3];

    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let vertex_shader_src = r#"
        #version 140

        in vec2 position;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * model * vec4(position, 0.0, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        out vec4 color;

        void main() {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let mut rng = rand::rng();
    let mut birds: Vec<Bird> = (0..NUM_BIRDS).map(|_| Bird::new(&mut rng)).collect();

    let mut delta_t: f32 = -0.5;

    #[allow(deprecated)] 
    let _ = event_loop.run(move |event, window_target| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => window_target.exit(),

                winit::event::WindowEvent::Resized(window_size) => {
                    display.resize(window_size.into());
                },

                winit::event::WindowEvent::RedrawRequested => {

                    let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
                    winit::event_loop::ControlFlow::WaitUntil(next_frame_time);

                    // --- Flocking update (parallel) ---
                    let perception_radius = 2.0;
                    let max_speed = 0.3;
                    let separation_weight = 1.5;
                    let alignment_weight = 1.0;
                    let cohesion_weight = 1.0;

                    let birds_snapshot = birds.clone();

                    birds.par_iter_mut().for_each(|bird| {
                        let mut separation = Vector3::zeros();
                        let mut alignment = Vector3::zeros();
                        let mut cohesion = Vector3::zeros();
                        let mut total = 0;

                        for other in &birds_snapshot {
                            let distance = (bird.position - other.position).norm();
                            if distance > 0.0 && distance < perception_radius {
                                // Separation
                                separation += (bird.position - other.position) / distance;

                                // Alignment
                                alignment += other.velocity;

                                // Cohesion
                                cohesion += other.position;

                                total += 1;
                            }
                        }

                        if total > 0 {
                            // Separation
                            separation /= total as f32;

                            // Alignment
                            alignment /= total as f32;
                            alignment -= bird.velocity;

                            // Cohesion
                            cohesion /= total as f32;
                            cohesion -= bird.position;
                        }

                        // Combine with weights
                        bird.acceleration =
                            separation_weight * separation +
                            alignment_weight * alignment +
                            cohesion_weight * cohesion;

                        // Velocity update
                        bird.velocity += bird.acceleration;
                        if bird.velocity.norm() > max_speed {
                            bird.velocity = bird.velocity.normalize() * max_speed;
                        }

                        // Position update
                        bird.position += bird.velocity;

                        bird.position = wraparound(bird.position);
                    });

                    // --- Rendering ---
                    let mut target = display.draw();
                    target.clear_color(0.0, 0.0, 0.0, 1.0);

                    let perspective = Perspective3::new(1.0, std::f32::consts::FRAC_PI_3, 0.1, 100.0);
                    let projection_matrix: [[f32; 4]; 4] = *perspective.as_matrix().as_ref();
                    let eye = Point3::new(0.0, 0.0, 20.0);
                    let look = Point3::origin();
                    let up = Vector3::y();
                    let view_matrix: [[f32; 4]; 4] = *Matrix4::look_at_rh(&eye, &look, &up).as_ref();

                    for bird in &birds {
                        let model_matrix = [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [bird.position.x, bird.position.y, bird.position.z, 1.0],
                        ];
                        let uniforms = uniform! {
                            model: model_matrix,
                            view: view_matrix,
                            projection: projection_matrix,
                        };
                        target.draw(&vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
                    }

                    target.finish().unwrap();
                },
                _ => (),
            },                
            winit::event::Event::AboutToWait => {
                window.request_redraw();
            },
            _ => (),
        };
    });
}
