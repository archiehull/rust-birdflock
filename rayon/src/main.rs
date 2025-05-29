#[macro_use]
extern crate glium;
extern crate winit;

use nalgebra::{Matrix4, Perspective3, Point3, Vector3}; // Add nalgebra for matrix calculations
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

const SHOW_VISUALS: bool = true;
const SHOW_TIMES: bool = true;
const SHOW_POSITIONS: bool = false;
const SHOWTIMES_EVERY: usize = 100;
const PRINT_EVERY: bool = false;

const SUMMARY_EVERY: usize = 1000;

const NUM_BIRDS: usize = 10000;

const POV_DISTANCE: f32 = 17.5;

const DIMENSIONS: f32 = 7.5;
const SPACE_MIN: f32 = -DIMENSIONS;
const SPACE_MAX: f32 = DIMENSIONS;

const SEPARATION_WEIGHT: f32 = 1.5;    // flock tightness
const ALIGNMENT_WEIGHT:  f32 = 2.0;    // movement coordination
const COHESION_WEIGHT:   f32 = 1.5;    // flock unification
const PERCEPTION_RADIUS: f32 = 1.9;    // flock size
const MAX_SPEED:         f32 = 0.125;
const MAX_FORCE:         f32 = 0.03;   // sharpness of movement



#[derive(Clone)]
struct Bird {
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    acceleration: Vector3<f32>,
}

impl Bird {
    // Create a new bird with random position and velocity
    fn new<R: Rng>(rng: &mut R) -> Self {
        Bird {
            position: Vector3::new(
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0)
            ),
            velocity: Vector3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0)
            ),
            acceleration: Vector3::zeros(),
        }
    }
}

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

fn limit_vec(v: Vector3<f32>, max: f32) -> Vector3<f32> {
    if v.norm() > max {
        v.normalize() * max
    } else {
        v
    }
}

fn main() {
    #[allow(unused_imports)]
    use glium::{glutin, Surface};

    let event_loop = glium::winit::event_loop::EventLoop::builder()
        .build()
        .expect("event loop building");
    let (window, display) = glium::backend::glutin::SimpleWindowBuilder::new()
        .with_title("Bird Flocking Simulation")
        .build(&event_loop);

    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
    }

    implement_vertex!(Vertex, position);

    // Create a triangle shape
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

    // bird colour
    let fragment_shader_src = r#"
        #version 140

        uniform float depth; // z position of the bird

        out vec4 color;

        void main() {
            // Map depth (e.g. -7.5 to 7.5) to [0,1]
            float t = clamp((depth + 7.5) / 15.0, 0.0, 1.0);
            // Example: from red (far) to white (close)
            vec3 near_col = vec3(1.0, 1.0, 1.0);   // white when close
            vec3 far_col = vec3(1.0, 0.2, 0.2);    // red when far
            vec3 bird_col = mix(far_col, near_col, 1.0 - t);
            color = vec4(bird_col, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    // Initialize birds with random positions and velocities
    let mut rng = rand::rng();
    let mut birds: Vec<Bird> = (0..NUM_BIRDS).map(|_| Bird::new(&mut rng)).collect();

    let mut step_count = 0;
    let mut total_steps = 0;
    let mut perf_start = Instant::now();
    let mut summary_start = Instant::now();

    let mut total_overhead_time = 0.0;
    let mut total_calc_time = 0.0;

    let mut cumulative_overhead_time = 0.0;
    let mut cumulative_calc_time = 0.0;

    println!("\n\nStarting simulation with {} birds using Rayon", NUM_BIRDS);
    if SHOW_VISUALS {
        println!("Visuals enabled.\n");
    } else {
        println!("Visuals disabled.\n");
    }

    #[allow(deprecated)] 
    let _ = event_loop.run(move |event, window_target| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => window_target.exit(),

                winit::event::WindowEvent::Resized(window_size) => {
                    display.resize(window_size.into());
                },

                winit::event::WindowEvent::RedrawRequested => {
                    if SHOW_TIMES && step_count == 0 {
                        perf_start = Instant::now();
                    }
    
                    if total_steps == 0 {
                        summary_start = Instant::now();
                    }
    
                    let step_start = Instant::now();

                    // --- Flocking update (parallel) ---
                    let birds_snapshot = birds.clone();

                    let calc_start = Instant::now();

                    birds.par_iter_mut().enumerate().for_each(|(i, bird)| {
                        let mut separation = Vector3::zeros();
                        let mut alignment = Vector3::zeros();
                        let mut cohesion = Vector3::zeros();
                        let mut total = 0;

                        for other in &birds_snapshot {
                            let distance = (bird.position - other.position).norm();
                            if distance > 0.0 && distance < PERCEPTION_RADIUS {

                                separation += (bird.position - other.position) / distance;
                                alignment += other.velocity;
                                cohesion += other.position;

                                total += 1;
                            }
                        }

                        if total > 0 {
                            // Separation
                            separation /= total as f32;
                            if separation.norm() > 0.0 {
                                separation = separation.normalize() * MAX_SPEED - bird.velocity;
                                separation = limit_vec(separation, MAX_FORCE);
                            }

                            // Alignment
                            alignment /= total as f32;
                            if alignment.norm() > 0.0 {
                                alignment = alignment.normalize() * MAX_SPEED - bird.velocity;
                                alignment = limit_vec(alignment, MAX_FORCE);
                            }

                            // Cohesion
                            cohesion /= total as f32;
                            cohesion = cohesion - bird.position;
                            if cohesion.norm() > 0.0 {
                                cohesion = cohesion.normalize() * MAX_SPEED - bird.velocity;
                                cohesion = limit_vec(cohesion, MAX_FORCE);
                            }
                        }
                        
                        // Combine with weights
                        bird.acceleration =
                            SEPARATION_WEIGHT * separation +
                            ALIGNMENT_WEIGHT * alignment +
                            COHESION_WEIGHT * cohesion;
                        
                        // Velocity update and limit speed
                        bird.velocity += bird.acceleration;
                        if bird.velocity.norm() > MAX_SPEED {
                            bird.velocity = bird.velocity.normalize() * MAX_SPEED;
                        }
                        
                        // Position update
                        bird.position += bird.velocity;
                        bird.position = wraparound(bird.position);

                        if SHOW_POSITIONS {
                            println!(
                                "Bird {}: pos={:?} vel={:?} sep={:?} ali={:?} coh={:?}",
                                i, bird.position, bird.velocity, separation, alignment, cohesion
                            );
                        }
                    });

                    let calc_time = calc_start.elapsed().as_secs_f64();
                    total_calc_time += calc_time;
                    cumulative_calc_time += calc_time;

                    // --- Rendering ---
                    if SHOW_VISUALS {
                        let mut target = display.draw();
                        target.clear_color(0.0, 0.0, 0.0, 1.0);

                        let perspective = Perspective3::new(1.0, std::f32::consts::FRAC_PI_3, 0.1, 100.0);
                        let projection_matrix: [[f32; 4]; 4] = *perspective.as_matrix().as_ref();
                        let eye = Point3::new(0.0, 0.0, POV_DISTANCE);
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
                                depth: bird.position.z, // Pass z position
                            };
                            target.draw(&vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
                        }

                        target.finish().unwrap();
                    }

                    let overhead_time = step_start.elapsed().as_secs_f64() - calc_time;
                    total_overhead_time += overhead_time;
                    cumulative_overhead_time += overhead_time;
                    
                    if SHOW_TIMES {
                        step_count += 1;
                        total_steps += 1;
    
                        // In the RedrawRequested event handler, replace the print sections:

                        if step_count % SHOWTIMES_EVERY == 0 && PRINT_EVERY {
                            let elapsed = perf_start.elapsed();
                            let avg_time_per_step = elapsed.as_secs_f64() / SHOWTIMES_EVERY as f64;
                            let fps = 1.0 / avg_time_per_step;
                            
                            let avg_calc_time = total_calc_time / SHOWTIMES_EVERY as f64;
                            let avg_overhead = total_overhead_time / SHOWTIMES_EVERY as f64;

                            println!(
                                "Simulated steps {}-{} in {:.3} seconds ({:.3} ms/step, {:.2} FPS)",
                                total_steps - SHOWTIMES_EVERY,
                                total_steps,
                                elapsed.as_secs_f64(),
                                avg_time_per_step * 1000.0,
                                fps
                            );
                            println!(
                                "Calculation: {:.3} ms | Overhead: {:.3} ms | Total: {:.3} ms",
                                avg_calc_time * 1000.0,
                                avg_overhead * 1000.0,
                                (avg_calc_time + avg_overhead) * 1000.0
                            );
                            
                            // Reset counters for the next batch
                            total_calc_time = 0.0;
                            total_overhead_time = 0.0;
                            perf_start = Instant::now();
                        }

                        if total_steps % SUMMARY_EVERY == 0 {
                            let summary_elapsed = summary_start.elapsed();
                            let avg_fps = SUMMARY_EVERY as f64 / summary_elapsed.as_secs_f64();

                            let avg_calc = (cumulative_calc_time / SUMMARY_EVERY as f64) * 1000.0;
                            let avg_overhead = (cumulative_overhead_time / SUMMARY_EVERY as f64) * 1000.0;

                            println!(
                                "\n\nSimulated {} steps in {:.3} seconds at {:.0} FPS",
                                SUMMARY_EVERY,
                                summary_elapsed.as_secs_f64(),
                                avg_fps
                            );
                            println!(
                                "Average Calculation: {:.3} ms | Average Overhead: {:.3} ms",
                                avg_calc,
                                avg_overhead
                            );
                            println!("\nSimulation complete. Exiting.");
                            window_target.exit();
                        }
                    }
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
