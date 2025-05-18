#[macro_use]
extern crate glium;
extern crate winit;

use nalgebra::{Matrix4, Perspective3, Point3, Vector3};
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use threadpool::ThreadPool;
use num_cpus;

const SHOW_VISUALS: bool = true;
const SHOW_TIMES: bool = true;
const SHOW_POSITIONS: bool = false;
const SHOWTIMES_EVERY: usize = 1000;

const NUM_BIRDS: usize = 750;
const POV_DISTANCE: f32 = 17.5;
const DIMENSIONS: f32 = 7.5;
const SPACE_MIN: f32 = -DIMENSIONS;
const SPACE_MAX: f32 = DIMENSIONS;

const SEPARATION_WEIGHT: f32 = 1.5;
const ALIGNMENT_WEIGHT:  f32 = 2.0;
const COHESION_WEIGHT:   f32 = 1.5;
const PERCEPTION_RADIUS: f32 = 1.9;
const MAX_SPEED:         f32 = 0.125;
const MAX_FORCE:         f32 = 0.03;

#[derive(Clone)]
struct Bird {
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    acceleration: Vector3<f32>,
}

impl Bird {
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

fn calculate_bird_update(
    index: usize,
    bird: &Bird,
    birds_snapshot: &[Bird],
) -> (Vector3<f32>, Vector3<f32>) {
    let mut separation = Vector3::zeros();
    let mut alignment = Vector3::zeros();
    let mut cohesion = Vector3::zeros();
    let mut total = 0;

    for other in birds_snapshot {
        let distance = (bird.position - other.position).norm();
        if distance > 0.0 && distance < PERCEPTION_RADIUS {
            separation += (bird.position - other.position) / distance;
            alignment += other.velocity;
            cohesion += other.position;
            total += 1;
        }
    }

    let mut acceleration = Vector3::zeros();

    if total > 0 {
        let mut sep_force = Vector3::zeros();
        separation /= total as f32;
        if separation.norm() > 0.0 {
            sep_force = separation.normalize() * MAX_SPEED - bird.velocity;
            sep_force = limit_vec(sep_force, MAX_FORCE);
        }

        let mut align_force = Vector3::zeros();
        alignment /= total as f32;
        if alignment.norm() > 0.0 {
            align_force = alignment.normalize() * MAX_SPEED - bird.velocity;
            align_force = limit_vec(align_force, MAX_FORCE);
        }

        let mut coh_force = Vector3::zeros();
        cohesion /= total as f32;
        cohesion = cohesion - bird.position;
        if cohesion.norm() > 0.0 {
            coh_force = cohesion.normalize() * MAX_SPEED - bird.velocity;
            coh_force = limit_vec(coh_force, MAX_FORCE);
        }

        acceleration =
            SEPARATION_WEIGHT * sep_force +
            ALIGNMENT_WEIGHT * align_force +
            COHESION_WEIGHT * coh_force;
    }

    let mut velocity = bird.velocity + acceleration;
    if velocity.norm() > MAX_SPEED {
        velocity = velocity.normalize() * MAX_SPEED;
    }

    let position = wraparound(bird.position + velocity);

    if SHOW_POSITIONS {
        println!(
            "Bird {}: pos={:?} vel={:?} sep={:?} ali={:?} coh={:?}",
            index, position, velocity, separation, alignment, cohesion
        );
    }

    (position, velocity)
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

        uniform float depth;

        out vec4 color;

        void main() {
            float t = clamp((depth + 7.5) / 15.0, 0.0, 1.0);
            vec3 near_col = vec3(1.0, 1.0, 1.0);
            vec3 far_col = vec3(1.0, 0.2, 0.2);
            vec3 bird_col = mix(far_col, near_col, 1.0 - t);
            color = vec4(bird_col, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let mut rng = rand::rng();
    let mut birds: Vec<Bird> = (0..NUM_BIRDS).map(|_| Bird::new(&mut rng)).collect();

    let num_threads = num_cpus::get();
    let pool = ThreadPool::new(num_threads);

    let mut step_count = 0;
    let mut perf_start = Instant::now();

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

                    // --- Flocking update with thread pool ---
                    let birds_snapshot = birds.clone();
                    let results = Arc::new(Mutex::new(vec![(Vector3::zeros(), Vector3::zeros()); NUM_BIRDS]));
                    let chunk_size = (NUM_BIRDS + num_threads - 1) / num_threads;
                    let num_tasks = (NUM_BIRDS + chunk_size - 1) / chunk_size;
                    let completed_count = Arc::new(Mutex::new(0));

                    for thread_id in 0..num_tasks {
                        let start = thread_id * chunk_size;
                        let end = (start + chunk_size).min(NUM_BIRDS);

                        let birds_snapshot = birds_snapshot.clone();
                        let results = Arc::clone(&results);
                        let completed_count = Arc::clone(&completed_count);

                        pool.execute(move || {
                            let mut local_results = Vec::new();

                            for i in start..end {
                                let update = calculate_bird_update(i, &birds_snapshot[i], &birds_snapshot);
                                local_results.push((i, update));
                            }

                            let mut results_guard = results.lock().unwrap();
                            for (i, update) in local_results {
                                results_guard[i] = update;
                            }

                            let mut count = completed_count.lock().unwrap();
                            *count += 1;
                        });
                    }

                    let wait_start = Instant::now();
                    while *completed_count.lock().unwrap() < num_tasks {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        if wait_start.elapsed().as_secs() > 5 {
                            println!("Warning: Tasks taking too long, continuing anyway");
                            break;
                        }
                    }

                    let results_guard = results.lock().unwrap();
                    for (i, &(position, velocity)) in results_guard.iter().enumerate() {
                        birds[i].position = position;
                        birds[i].velocity = velocity;
                        birds[i].acceleration = Vector3::zeros();
                    }

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
                                depth: bird.position.z,
                            };
                            target.draw(&vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
                        }

                        target.finish().unwrap();
                    }

                    if SHOW_TIMES {
                        step_count += 1;
                        if step_count >= SHOWTIMES_EVERY {
                            let elapsed = perf_start.elapsed();
                            println!(
                                "Simulated {} steps in {:.3} seconds ({:.3} ms/step)",
                                SHOWTIMES_EVERY,
                                elapsed.as_secs_f64(),
                                elapsed.as_secs_f64() * 1000.0 / SHOWTIMES_EVERY as f64
                            );
                            step_count = 0;
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