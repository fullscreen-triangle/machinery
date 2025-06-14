use machinery::examples::run_all_demonstrations;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🏥 Machinery Temporal Dynamics Demo");
    println!("=====================================");
    println!();
    println!("This demo showcases the core temporal dynamics concepts:");
    println!("- Temporal data validity and decay");
    println!("- Dynamic medium effects (system changing while measuring)");
    println!("- Context-aware prediction vs naive approaches");
    println!("- Measurement latency and biological delays");
    println!();

    // Run all demonstrations
    run_all_demonstrations()?;

    println!("\n🎯 Demo Complete!");
    println!("This implementation demonstrates how Machinery handles");
    println!("the complex temporal nature of biological systems.");
    
    Ok(())
} 