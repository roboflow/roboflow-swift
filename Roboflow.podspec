Pod::Spec.new do |spec|
  spec.name               = "Roboflow"
  spec.version            = "1.2.5"
  spec.platform           = :ios, '15.2'
  spec.ios.deployment_target = '15.2'
  spec.summary            = "A framework for interfacing with Roboflow"
  spec.description        = "A framework for interfacing with hosted computer vision models on Roboflow.com"
  spec.homepage           = "https://www.roboflow.com"
  spec.documentation_url  = "https://docs.roboflow.com/developer/ios-sdk"
  spec.license            = { :type => 'Apache', :text => 'See LICENSE at https://roboflow.com' }
  spec.author             = { "Roboflow" => "hello@roboflow.com" } 
  spec.swift_versions     = ['5.3']
  spec.source             = { :git => 'https://github.com/roboflow/roboflow-swift.git', :tag => "#{spec.version}" }
  spec.source_files       = 'Sources/**/*'
end
